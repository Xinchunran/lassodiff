"""Validated LassoPred JSON--PDB dataset stored in LMDB.

The legacy loader required exactly ``min1.pdb`` and ``relax1.pdb``.  This
module instead treats every available ``min[1-3]``/``relax[1-3]`` file as a
label conformer, records every rejected record, and keeps topology candidates
explicit.  LMDB is deliberately mandatory: silently falling back to pickle
would make multi-worker and multi-rank data access non-reproducible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
import random
import re
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .topology_targets import record_has_topology_target, select_topology_target

from lassodiff.pdb_utils import parse_pdb_residues, process_structure_backbone
from lassodiff.seq_encoder import seq_to_aa_ids

LASSOPRED_SCHEMA_VERSION = 1
ESM_CACHE_SCHEMA_VERSION = 1
_MANIFEST_KEY = b"__lassodiff_manifest__"
_RECORD_PREFIX = b"sample/"


class DatasetSchemaError(RuntimeError):
    """Raised for a malformed or incompatible LMDB dataset/cache."""


def require_lmdb():
    try:
        import lmdb
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise RuntimeError(
            "LMDB support is required for LassoDiff V2 data. Install `lmdb` "
            "in the PyTorch training environment; no storage fallback is used."
        ) from exc
    return lmdb


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _canonical_sequence(value: Any) -> str:
    sequence = "".join(str(value or "").upper().split())
    if not sequence or any(c not in "ACDEFGHIKLMNPQRSTVWY" for c in sequence):
        raise ValueError("Core_Sequence is absent or not canonical amino-acid text")
    return sequence


def _pdb_sequence_compatible(expected: str, observed: str) -> bool:
    """Accept PDB ambiguity codes while rejecting unambiguous mismatches."""
    if len(expected) != len(observed):
        return False
    compatible = {"B": set("DN"), "Z": set("EQ"), "X": set(expected), "U": {"C"}, "O": {"K"}}
    for want, got in zip(expected, observed):
        if got == want:
            continue
        if got in compatible and want in compatible[got]:
            continue
        return False
    return True


def _acceptor_type_and_index(pdb_path: Path, iso_index0: int) -> Tuple[int, str]:
    residues = parse_pdb_residues(str(pdb_path))
    if not (0 <= iso_index0 < len(residues)):
        raise ValueError(f"iso_acceptor_index={iso_index0} is outside PDB length {len(residues)}")
    resname, _key, atoms = residues[iso_index0]
    # Amber/minimisation exports frequently write the reactive residue as ASX
    # or GLX.  Atom names, rather than the ambiguous residue label, decide the
    # chemistry; this is distinct from accepting an arbitrary nearby residue.
    if resname in ("ASP", "ASH", "ASX") and "CG" in atoms:
        return iso_index0, "ASP"
    if resname in ("GLU", "GLH", "GLX") and "CD" in atoms:
        return iso_index0, "GLU"
    # The metadata index can be off by one in older exports.  Do not repair it
    # silently; a nearby chemically valid acceptor is unambiguous enough to log.
    nearby = []
    for i, (name, _res, atom_dict) in enumerate(residues):
        if name in ("ASP", "ASH", "ASX") and "CG" in atom_dict:
            nearby.append((abs(i - iso_index0), i, "ASP"))
        elif name in ("GLU", "GLH", "GLX") and "CD" in atom_dict:
            nearby.append((abs(i - iso_index0), i, "GLU"))
    if not nearby:
        raise ValueError("iso acceptor is not ASP/GLU and no compatible acceptor exists")
    distance, resolved, kind = min(nearby)
    if distance > 1:
        raise ValueError("metadata isopeptide position does not resolve to a nearby ASP/GLU")
    return resolved, kind


def _topology_candidates(meta: Mapping[str, Any], length: int, acceptor_index: int, acceptor_type: str) -> List[Dict[str, Any]]:
    ring_length = int(meta.get("Ring_Length") or 0)
    if not (1 <= ring_length <= length):
        raise ValueError(f"invalid Ring_Length={ring_length} for length={length}")
    k = ring_length - 1
    plug_fields = ("Upper_Plug_1", "Upper_Plug_2", "Upper_Plug_3")
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for rank, field in enumerate(plug_fields):
        raw = meta.get(field)
        if raw is None or raw == "":
            continue
        p = int(raw) - 1  # JSON labels are residue numbers (one based).
        if not (k < p < length) or p in seen:
            continue
        seen.add(p)
        candidates.append({
            "k": k,
            "p": p,
            "acceptor_index": acceptor_index,
            "acceptor_type": acceptor_type,
            "prior": 0.0,
            "rank": rank + 1,
        })
    if not candidates:
        # Loop_Length_1 is the historical fallback, but remains explicit in
        # the resulting record rather than hidden in the loader.
        loop = int(meta.get("Loop_Length_1") or 0)
        p = k + loop
        if not (k < p < length):
            raise ValueError("no valid plug candidate or Loop_Length_1 fallback")
        candidates.append({
            "k": k, "p": p, "acceptor_index": acceptor_index,
            "acceptor_type": acceptor_type, "prior": 0.0, "rank": 1,
        })
    prior = 1.0 / len(candidates)
    for candidate in candidates:
        candidate["prior"] = prior
    return candidates


def _read_conformers(
    entry: Path, sequence: str, iso_index0: int, acceptor_type: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    conformers: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    for kind in ("min", "relax"):
        for rank in range(1, 4):
            path = entry / f"{kind}{rank}.pdb"
            if not path.is_file():
                continue
            try:
                processed = process_structure_backbone(
                    str(path), iso_acceptor_index=iso_index0 + 1, iso_acceptor_type=acceptor_type
                )
                observed = str(processed["seq"])
                if not _pdb_sequence_compatible(sequence, observed):
                    raise ValueError(
                        f"sequence conflicts with Core_Sequence: {observed} vs {sequence}"
                    )
                coords = processed["coords"]
                atom_mask = processed["mask"]
                if coords.shape != (len(sequence), 7, 3) or atom_mask.shape != (len(sequence), 7):
                    raise ValueError(f"invalid processed shape {tuple(coords.shape)}")
                if not bool(atom_mask[:, :4].all()):
                    raise ValueError("missing backbone atoms")
                if not bool(torch.isfinite(coords[atom_mask]).all()):
                    raise ValueError("contains non-finite coordinates")
                conformers.append({
                    "name": path.stem,
                    "coords": coords.contiguous(),
                    "atom_mask": atom_mask.contiguous(),
                })
            except (ValueError, OSError) as exc:
                rejected.append({"conformer": path.name, "reason": str(exc)})
    if not conformers:
        detail = "; ".join(f"{item['conformer']}: {item['reason']}" for item in rejected[:3])
        raise ValueError(f"no valid min[1-3]/relax[1-3] PDB label found ({detail})")
    return conformers, rejected


def build_record(meta: Mapping[str, Any], structure_dir: Path) -> Dict[str, Any]:
    record_id = str(meta.get("LP_ID") or "")
    if not record_id:
        raise ValueError("metadata has no LP_ID")
    sequence = _canonical_sequence(meta.get("Core_Sequence"))
    declared_length = meta.get("Core_Length")
    if declared_length is not None and int(declared_length) != len(sequence):
        raise ValueError(f"Core_Length={declared_length} differs from Core_Sequence length={len(sequence)}")
    entry = structure_dir / record_id
    if not entry.is_dir():
        raise ValueError(f"structure directory does not exist: {entry}")
    iso_raw = meta.get("Isopeptide")
    if iso_raw is None or iso_raw == "":
        raise ValueError("metadata has no Isopeptide label")
    iso_index0 = int(iso_raw) - 1
    first_pdb = next((p for p in (entry / "min1.pdb", entry / "min2.pdb", entry / "min3.pdb", entry / "relax1.pdb", entry / "relax2.pdb", entry / "relax3.pdb") if p.is_file()), None)
    if first_pdb is None:
        raise ValueError("no PDB label found")
    iso_index0, acceptor_type = _acceptor_type_and_index(first_pdb, iso_index0)
    conformers, conformer_rejects = _read_conformers(entry, sequence, iso_index0, acceptor_type)
    candidates = _topology_candidates(meta, len(sequence), iso_index0, acceptor_type)
    return {
        "record_id": record_id,
        "sequence": sequence,
        "sequence_hash": sequence_hash(sequence),
        "length": len(sequence),
        "iso_acceptor_index": iso_index0,
        "iso_acceptor_type": acceptor_type,
        "candidates": candidates,
        "conformers": conformers,
        "conformer_rejects": conformer_rejects,
        "source": {"metadata_json": record_id, "structure_dir": str(entry)},
    }


def build_lassopred_lmdb(
    metadata_json: str | Path,
    structure_dir: str | Path,
    output: str | Path,
    reject_report: str | Path | None = None,
    map_size: int = 32 * 1024**3,
    overwrite: bool = False,
    verify_only: bool = False,
) -> Dict[str, Any]:
    """Build a deterministic validated LMDB and return its manifest.

    ``verify_only`` performs exactly the same validation without writing LMDB;
    it is useful on login nodes before requesting GPU time.
    """
    metadata_json = Path(metadata_json)
    structure_dir = Path(structure_dir)
    output = Path(output)
    reject_path = Path(reject_report) if reject_report else output.with_suffix(".rejects.jsonl")
    rows = json.loads(metadata_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("metadata JSON must be a list")
    records, rejects, conformer_rejects = [], [], []
    seen = set()
    for meta in sorted(rows, key=lambda item: str(item.get("LP_ID") or "")):
        record_id = str(meta.get("LP_ID") or "")
        if not record_id or record_id in seen:
            rejects.append({"record_id": record_id, "reason": "missing_or_duplicate_LP_ID"})
            continue
        seen.add(record_id)
        try:
            record = build_record(meta, structure_dir)
            records.append(record)
            conformer_rejects.extend(
                {"record_id": record_id, "severity": "conformer", **item}
                for item in record.get("conformer_rejects", [])
            )
        except (ValueError, OSError) as exc:
            rejects.append({"record_id": record_id, "reason": str(exc)})
    manifest = {
        "schema_version": LASSOPRED_SCHEMA_VERSION,
        "dataset": "lassopred_json_pdb",
        "metadata_json": str(metadata_json.resolve()),
        "structure_dir": str(structure_dir.resolve()),
        "record_count": len(records),
        "reject_count": len(rejects),
        "conformer_reject_count": len(conformer_rejects),
        "record_ids": [record["record_id"] for record in records],
    }
    reject_path.parent.mkdir(parents=True, exist_ok=True)
    report_rows = rejects + conformer_rejects
    reject_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in report_rows), encoding="utf-8")
    if verify_only:
        return manifest
    lmdb = require_lmdb()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty LMDB path: {output}")
    output.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output), map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False)
    with env.begin(write=True) as txn:
        txn.put(_MANIFEST_KEY, pickle.dumps(manifest, protocol=pickle.HIGHEST_PROTOCOL), overwrite=True)
        for record in records:
            key = _RECORD_PREFIX + record["record_id"].encode("utf-8")
            txn.put(key, pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL), overwrite=True)
    env.sync()
    env.close()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


class LassoPredLMDBDataset(Dataset):
    """Lazy LMDB reader safe to use from DataLoader workers and DDP ranks."""

    def __init__(
        self,
        path: str | Path,
        record_ids: Optional[Sequence[str]] = None,
        seed: int = 0,
        esm_cache_path: str | Path | None = None,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        esm_repr_layer: int = 33,
        target_policy: str = "all",
    ):
        self.path = Path(path)
        self.seed = int(seed)
        self.epoch = 0
        if target_policy not in {"all", "topology_valid"}:
            raise ValueError("target_policy must be all or topology_valid")
        self.target_policy = target_policy
        self._env = None
        self.esm_cache = (
            ESMEmbeddingCache(esm_cache_path, esm_model_name, esm_repr_layer)
            if esm_cache_path is not None else None
        )
        lmdb = require_lmdb()
        env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False, subdir=True, max_readers=512)
        with env.begin() as txn:
            payload = txn.get(_MANIFEST_KEY)
        env.close()
        if payload is None:
            raise DatasetSchemaError("LMDB is missing LassoDiff manifest")
        self.manifest = pickle.loads(payload)
        if self.manifest.get("schema_version") != LASSOPRED_SCHEMA_VERSION:
            raise DatasetSchemaError(
                f"dataset schema={self.manifest.get('schema_version')} != {LASSOPRED_SCHEMA_VERSION}"
            )
        known = set(self.manifest["record_ids"])
        self.record_ids = list(record_ids) if record_ids is not None else list(self.manifest["record_ids"])
        missing = set(self.record_ids) - known
        if missing:
            raise DatasetSchemaError(f"split references records not in LMDB: {sorted(missing)[:3]}")
        self.source_record_count = len(self.record_ids)
        if self.target_policy == "topology_valid":
            qualified = []
            scan = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False, subdir=True, max_readers=512)
            with scan.begin() as txn:
                for record_id in self.record_ids:
                    payload = txn.get(_RECORD_PREFIX + record_id.encode("utf-8"))
                    if payload is None:
                        raise DatasetSchemaError(f"missing LMDB record {record_id}")
                    if record_has_topology_target(pickle.loads(payload)):
                        qualified.append(record_id)
            scan.close()
            self.record_ids = qualified
            if not self.record_ids:
                raise DatasetSchemaError("topology_valid policy removed every record")
        self.qualified_record_count = len(self.record_ids)

    def _open(self):
        if self._env is None:
            self._env = require_lmdb().open(
                str(self.path), readonly=True, lock=False, readahead=False, subdir=True, max_readers=512
            )
        return self._env

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record_id = self.record_ids[index]
        with self._open().begin() as txn:
            payload = txn.get(_RECORD_PREFIX + record_id.encode("utf-8"))
        if payload is None:
            raise DatasetSchemaError(f"missing LMDB record {record_id}")
        record = pickle.loads(payload)
        # min1/relax1, min2/relax2 and min3/relax3 are MD/template labels for
        # candidate ranks 1, 2 and 3 respectively.  A single randomly selected
        # structure must never be copied across candidates: that destroys the
        # topology--structure correspondence required by the V2 objective.
        conformers_by_rank: Dict[int, List[Dict[str, Any]]] = {}
        for conformer in record["conformers"]:
            match = re.fullmatch(r"(?:min|relax)([1-3])", str(conformer["name"]))
            if match:
                conformers_by_rank.setdefault(int(match.group(1)), []).append(conformer)
        targets = []
        target_valid = []
        for candidate in record["candidates"]:
            rank = int(candidate["rank"])
            available = conformers_by_rank.get(rank, [])
            if not available:
                raise DatasetSchemaError(
                    f"record {record_id} candidate rank {rank} has no matching min/relax target"
                )
            # Deterministic per candidate, epoch and record.  This lets train
            # alternate min/relax labels without making validation unstable.
            digest = hashlib.sha256(
                f"{self.seed}:{self.epoch}:{record_id}:candidate:{rank}".encode()
            ).hexdigest()
            deterministic_index = int(digest, 16)
            if self.target_policy == "topology_valid":
                target, valid = select_topology_target(
                    available, candidate, deterministic_index=deterministic_index,
                )
            else:
                target, valid = available[deterministic_index % len(available)], True
            targets.append(target)
            target_valid.append(bool(valid))
        item = {
            "record_id": record_id,
            "sequence": record["sequence"],
            "aa_ids": seq_to_aa_ids(record["sequence"]),
            "coords": torch.stack([target["coords"].float() for target in targets]),
            "atom_mask": torch.stack([target["atom_mask"].bool() for target in targets]),
            "target_names": [str(target["name"]) for target in targets],
            "target_valid": torch.tensor(target_valid, dtype=torch.bool),
            "candidates": record["candidates"],
            "iso_acceptor_index": int(record["iso_acceptor_index"]),
            "iso_acceptor_type": record["iso_acceptor_type"],
        }
        if self.esm_cache is not None:
            item["esm_residue"] = self.esm_cache.get(record["sequence"])
        return item


def collate_lassopred_v2(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length peptides and variable candidate counts without loss."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    B = len(batch)
    L = max(int(item["aa_ids"].numel()) for item in batch)
    M = max(len(item["candidates"]) for item in batch)
    A = int(batch[0]["coords"].shape[2])
    aa_ids = torch.full((B, L), 20, dtype=torch.long)
    token_mask = torch.zeros((B, L), dtype=torch.bool)
    coords = torch.zeros((B, M, L, A, 3), dtype=torch.float32)
    atom_mask = torch.zeros((B, M, L, A), dtype=torch.bool)
    esm_dim = int(batch[0]["esm_residue"].shape[-1]) if "esm_residue" in batch[0] else None
    esm_residue = torch.zeros((B, L, esm_dim), dtype=torch.float32) if esm_dim is not None else None
    k = torch.zeros((B, M), dtype=torch.long)
    p = torch.zeros((B, M), dtype=torch.long)
    acceptor_index = torch.zeros((B, M), dtype=torch.long)
    prior = torch.zeros((B, M), dtype=torch.float32)
    candidate_mask = torch.zeros((B, M), dtype=torch.bool)
    target_valid = torch.zeros((B, M), dtype=torch.bool)
    names, seqs = [], []
    target_names: List[List[Optional[str]]] = []
    for b, item in enumerate(batch):
        length = int(item["aa_ids"].numel())
        aa_ids[b, :length] = item["aa_ids"]
        token_mask[b, :length] = True
        item_candidates = len(item["candidates"])
        expected_coords = (item_candidates, length, A, 3)
        expected_mask = (item_candidates, length, A)
        if tuple(item["coords"].shape) != expected_coords or tuple(item["atom_mask"].shape) != expected_mask:
            raise DatasetSchemaError(
                f"candidate target shape mismatch: expected {expected_coords}/{expected_mask}, "
                f"got {tuple(item['coords'].shape)}/{tuple(item['atom_mask'].shape)}"
            )
        coords[b, :item_candidates, :length] = item["coords"]
        atom_mask[b, :item_candidates, :length] = item["atom_mask"]
        if esm_residue is not None:
            embedding = item.get("esm_residue")
            if embedding is None or tuple(embedding.shape) != (length, esm_dim):
                raise DatasetSchemaError("batch has missing or inconsistent ESM residue embedding")
            esm_residue[b, :length] = embedding
        for m, candidate in enumerate(item["candidates"]):
            k[b, m] = int(candidate["k"])
            p[b, m] = int(candidate["p"])
            acceptor_index[b, m] = int(candidate["acceptor_index"])
            prior[b, m] = float(candidate["prior"])
            candidate_mask[b, m] = True
            item_target_valid = item.get("target_valid")
            target_valid[b, m] = True if item_target_valid is None else bool(item_target_valid[m])
        names.append(str(item["record_id"]))
        seqs.append(str(item["sequence"]))
        row_target_names: List[Optional[str]] = [None] * M
        for m, target_name in enumerate(item["target_names"]):
            row_target_names[m] = str(target_name)
        target_names.append(row_target_names)
    # A filtered/sampled candidate set must remain a probability distribution;
    # this is important when a batch pads or deliberately truncates candidates.
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(prior.dtype).tiny)
    output = {
        "names": names, "seqs": seqs, "target_names": target_names,
        "aa_ids": aa_ids, "token_mask": token_mask, "coords": coords,
        "atom_mask": atom_mask, "k": k, "p": p,
        "acceptor_index": acceptor_index, "candidate_prior": prior,
        "candidate_mask": candidate_mask,
        "target_valid": target_valid,
    }
    if esm_residue is not None:
        output["esm_residue"] = esm_residue
    return output


def make_split_manifest(
    record_ids: Sequence[str],
    seed: int = 0,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> Dict[str, Any]:
    """Deterministic record-level split; a record's conformers cannot leak."""
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("invalid split fractions")
    ids = sorted(record_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_fraction))
    n_val = int(round(len(ids) * val_fraction))
    return {
        "version": 1, "seed": int(seed), "val_fraction": val_fraction, "test_fraction": test_fraction,
        "test": sorted(ids[:n_test]), "val": sorted(ids[n_test:n_test + n_val]),
        "train": sorted(ids[n_test + n_val:]),
    }


class ESMEmbeddingCache:
    """Read-only sequence-hash keyed ESM residue embedding cache.

    The cache's model identity and representation layer are part of its
    manifest.  A different ESM model is an error, never a cache hit.
    """

    def __init__(self, path: str | Path, model_name: str, repr_layer: int):
        self.path = Path(path)
        self.model_name = model_name
        self.repr_layer = int(repr_layer)
        self._env = None
        lmdb = require_lmdb()
        env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False, subdir=True, max_readers=512)
        with env.begin() as txn:
            raw = txn.get(_MANIFEST_KEY)
        env.close()
        if raw is None:
            raise DatasetSchemaError("ESM cache has no manifest")
        self.manifest = pickle.loads(raw)
        expected = (ESM_CACHE_SCHEMA_VERSION, model_name, int(repr_layer))
        got = (self.manifest.get("schema_version"), self.manifest.get("model_name"), self.manifest.get("repr_layer"))
        if got != expected:
            raise DatasetSchemaError(f"ESM cache mismatch: expected={expected}, got={got}")

    def _open(self):
        if self._env is None:
            self._env = require_lmdb().open(
                str(self.path), readonly=True, lock=False, readahead=False, subdir=True, max_readers=512
            )
        return self._env

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def get(self, sequence: str) -> torch.Tensor:
        key = _RECORD_PREFIX + sequence_hash(sequence).encode("ascii")
        with self._open().begin() as txn:
            raw = txn.get(key)
        if raw is None:
            raise KeyError(f"ESM cache miss for sequence sha256={sequence_hash(sequence)}")
        record = pickle.loads(raw)
        if record.get("sequence") != sequence:
            raise DatasetSchemaError("ESM cache hash collision or corrupted entry")
        embedding = record.get("embedding")
        if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2 or embedding.shape[0] != len(sequence):
            raise DatasetSchemaError("malformed ESM residue embedding")
        return embedding.float()


def write_esm_cache(
    output: str | Path,
    model_name: str,
    repr_layer: int,
    embeddings: Iterable[Tuple[str, torch.Tensor]],
    map_size: int = 64 * 1024**3,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Persist precomputed frozen ESM embeddings; used by the cache script."""
    output = Path(output)
    lmdb = require_lmdb()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty ESM cache: {output}")
    output.mkdir(parents=True, exist_ok=True)
    unique: Dict[str, torch.Tensor] = {}
    for sequence, embedding in embeddings:
        if sequence in unique:
            continue
        if embedding.ndim != 2 or embedding.shape[0] != len(sequence):
            raise ValueError(f"invalid embedding shape {tuple(embedding.shape)} for sequence length {len(sequence)}")
        unique[sequence] = embedding.detach().cpu().to(dtype=torch.float16).contiguous()
    manifest = {
        "schema_version": ESM_CACHE_SCHEMA_VERSION,
        "model_name": model_name,
        "repr_layer": int(repr_layer),
        "sequence_count": len(unique),
        "dtype": "float16",
    }
    env = lmdb.open(str(output), map_size=map_size, subdir=True, lock=True, readahead=False, meminit=False)
    with env.begin(write=True) as txn:
        txn.put(_MANIFEST_KEY, pickle.dumps(manifest, protocol=pickle.HIGHEST_PROTOCOL), overwrite=True)
        for sequence, embedding in unique.items():
            record = {"sequence": sequence, "embedding": embedding}
            txn.put(_RECORD_PREFIX + sequence_hash(sequence).encode("ascii"), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
    env.sync()
    env.close()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
