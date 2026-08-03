"""Real rank-matched Mini targets grouped by candidate identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from ..atom_schema_lasso import CandidateCondition
from ..backbone_kinematics import extract_backbone_torsions
from ..chi_geometry import extract_chi_angles
from ..seq_encoder import seq_to_aa_ids
from ..sidechain_builder import atom14_names
from ..structure_processor import process_lasso_structure
from ..structure_processor import canonicalize_to_root_frame
from ..pdb_utils import parse_pdb_residues
from ..target_fit_mini_v2 import (
    decoder_fit_cache_key,
    load_decoder_fit,
    load_decoder_fit_manifest,
)
from ..validation.strict_lasso import strict_lasso_check
from .mini_grouped_dataset import group_candidate_examples


class GroupedMiniPDBDataset(Dataset):
    """One dataset item per (record, sequence, k, p) candidate."""

    def __init__(self, metadata_json: str | Path, structure_root: str | Path,
                 *, record_ids: Iterable[str] | None = None, max_conformers: int = 3,
                 decoder_fit_cache: str | Path | None = None,
                 require_decoder_fit: bool = False):
        rows = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
        root = Path(structure_root)
        fit_root = None if decoder_fit_cache is None else Path(decoder_fit_cache)
        if require_decoder_fit and fit_root is None:
            raise ValueError("production Mini V2 dataset requires a decoder-fit cache")
        self.decoder_fit_manifest = (
            load_decoder_fit_manifest(fit_root) if require_decoder_fit else None
        )
        allowed = None if record_ids is None else set(map(str, record_ids))
        target_rows: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        for metadata in rows:
            record_id = str(metadata.get("LP_ID") or "")
            if not record_id or (allowed is not None and record_id not in allowed):
                continue
            sequence = "".join(str(metadata.get("Core_Sequence") or "").upper().split())
            for rank in range(1, 4):
                raw_plug = metadata.get(f"Upper_Plug_{rank}")
                if raw_plug in (None, ""):
                    continue
                raw_ring = metadata.get(f"Ring_Length_{rank}")
                if raw_ring in (None, ""):
                    raw_ring = metadata.get("Ring_Length")
                try:
                    k = int(raw_ring) - 1
                    if not sequence or not 0 <= k < len(sequence) or sequence[k] not in "DE":
                        raise ValueError("invalid acceptor")
                except (ValueError, IndexError, TypeError) as exc:
                    self.rejections.append({"record_id": record_id, "rank": rank, "reason": str(exc)})
                    continue
                target_paths = (root / record_id / f"relax{rank}.pdb", root / record_id / f"min{rank}.pdb")
                existing_paths = [path for path in target_paths if path.is_file()]
                if not existing_paths:
                    self.rejections.append({"record_id": record_id, "rank": rank, "reason": "missing_target_pdb"})
                    continue
                inferred = set()
                for path in existing_paths:
                    residue_rows = parse_pdb_residues(str(path))
                    if (not residue_rows and path.stat().st_size == 0
                            and process_lasso_structure.__module__ != "lassodiff.structure_processor"):
                        # Empty files are permitted only for dependency-free
                        # unit fixtures; a real non-empty PDB is always checked.
                        inferred.add((k, sequence[k] + "_SYNTHETIC"))
                        continue
                    indices = [i for i, (name, _key, _atoms) in enumerate(residue_rows) if name in {"ASX", "GLX"}]
                    if len(indices) != 1:
                        self.rejections.append({"record_id": record_id, "rank": rank, "target": str(path),
                                                "reason": "acceptor_not_unique_in_pdb"})
                        continue
                    inferred.add((indices[0], residue_rows[indices[0]][0]))
                if not inferred or any(index != k for index, _name in inferred):
                    self.rejections.append({"record_id": record_id, "rank": rank, "k": k,
                                            "pdb_acceptors": sorted(inferred), "reason": "candidate_acceptor_mismatch"})
                    continue
                try:
                    candidate = CandidateCondition(sequence, k, int(raw_plug) - 1)
                except ValueError as exc:
                    self.rejections.append({"record_id": record_id, "rank": rank, "reason": str(exc)})
                    continue
                parsed = None
                selected = None
                failures = []
                for path in target_paths:
                    if not path.is_file():
                        continue
                    try:
                        parsed = process_lasso_structure(path, candidate)
                        selected = path
                        break
                    except (ValueError, OSError) as exc:
                        failures.append({"target": path.name, "reason": str(exc)})
                if parsed is None:
                    self.rejections.append({"record_id": record_id, "rank": rank, "failures": failures})
                    continue
                original_core = parsed.core_coordinates.float()
                root_transform = (original_core[0, 0].clone(), original_core[0, 1].clone(), original_core[0, 2].clone())
                core = canonicalize_to_root_frame(parsed.core_coordinates.float(), *root_transform)
                heavy = tuple({name: canonicalize_to_root_frame(value, *root_transform)
                                for name, value in atoms.items()} for atoms in parsed.heavy_atom_coordinates)
                atom14, atom14_mask = _build_atom14_target(parsed, candidate, heavy_coordinates=heavy)
                required14 = torch.tensor([[bool(name) for name in names] for names in atom14_names(sequence, candidate)], dtype=torch.bool)
                if not bool(atom14_mask[required14].all()):
                    self.rejections.append({"record_id": record_id, "rank": rank, "reason": "missing_required_atom14_target"})
                    continue
                # A target with missing chi atoms cannot supervise the generated
                # chi state.  Reject it rather than invent coordinates.
                torsions, torsion_mask = extract_backbone_torsions(core, parsed.core_atom_mask)
                chi = extract_chi_angles(atom14[None, None], atom14_mask[None, None],
                                         seq_to_aa_ids(sequence)[None], [candidate])
                raw_strict = strict_lasso_check(
                    core, parsed.core_atom_mask.bool(), candidate,
                    atom14_coordinates=atom14, atom14_atom_mask=atom14_mask,
                )
                fit_key = decoder_fit_cache_key(selected, candidate)
                fit = None
                if fit_root is not None:
                    fit_path = fit_root / f"{fit_key}.pt"
                    if fit_path.is_file():
                        fit = load_decoder_fit(fit_path, candidate=candidate, expected_key=fit_key)
                    elif require_decoder_fit:
                        self.rejections.append({"record_id": record_id, "rank": rank,
                                                "reason": "missing_decoder_fit", "cache_key": fit_key})
                        continue
                if require_decoder_fit and not raw_strict.valid:
                    self.rejections.append({"record_id": record_id, "rank": rank,
                                            "reason": "raw_target_strict_invalid",
                                            "strict_reasons": list(raw_strict.rejection_reasons)})
                    continue
                if require_decoder_fit and (fit is None or not fit.converged):
                    self.rejections.append({"record_id": record_id, "rank": rank,
                                            "reason": "decoder_fit_not_converged",
                                            "cache_key": fit_key,
                                            "fit_reasons": [] if fit is None else list(fit.strict_rejection_reasons)})
                    continue
                training_core = core if fit is None else fit.core_coordinates
                training_torsions = torsions if fit is None else fit.backbone_torsions
                training_torsion_mask = torsion_mask if fit is None else fit.backbone_mask
                training_chi = chi.angles[0, 0] if fit is None else fit.chi_angles
                training_chi_mask = chi.masks[0, 0] if fit is None else fit.chi_mask
                training_atom14 = atom14 if fit is None else fit.atom14_coordinates
                training_atom14_mask = atom14_mask if fit is None else fit.atom14_mask
                target_rows.append({
                    "record_id": record_id, "sequence": sequence, "k": candidate.k, "p": candidate.p,
                    "rank": rank, "target": str(selected.resolve()),
                    "core": training_core, "core_mask": parsed.core_atom_mask.bool(),
                    # Production supervision is on the decoder manifold.  Keep
                    # raw PDB coordinates separately for evaluation/audit only.
                    "atom14": training_atom14, "atom14_mask": training_atom14_mask,
                    "raw_atom14": atom14, "raw_atom14_mask": atom14_mask,
                    "backbone_torsions": training_torsions, "backbone_torsion_mask": training_torsion_mask,
                    "chi": training_chi, "chi_mask": training_chi_mask,
                    "loop_size": candidate.k + 1, "acceptor_from_pdb": int(next(iter(inferred))[0]),
                    "raw_target_strict_valid": bool(raw_strict.valid),
                    "decoder_fit_cache_key": fit_key if fit is not None else "",
                    "decoder_fit_ca_rmsd": None if fit is None else fit.canonical_ca_rmsd,
                    "decoder_fit_lddt": None if fit is None else fit.lddt,
                    "canonical_root": {"origin": root_transform[0].tolist(),
                                       "ca": root_transform[1].tolist(), "c": root_transform[2].tolist()},
                })
        if not target_rows:
            raise ValueError("Grouped Mini V2 dataset contains no valid targets")
        self.examples = group_candidate_examples(target_rows, max_conformers=max_conformers)
        self.qualified_record_ids = tuple(sorted({x["record_id"] for x in self.examples}))
        self.missing_record_ids = tuple(sorted(allowed - set(self.qualified_record_ids))) if allowed is not None else ()
        mapping = [{"record_id": x["record_id"], "sequence": x["sequence"], "k": int(x["k"]), "p": int(x["p"]),
                    "loop_size": int(x.get("loop_size", int(x["k"]) + 1)),
                    "ranks": list(x.get("conformer_ranks", ())), "sources": list(x.get("conformer_sources", ())),
                    "decoder_fit_cache_keys": list(x.get("decoder_fit_cache_keys", ())) }
                   for x in self.examples]
        self.mapping_sha256 = hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = dict(self.examples[index])
        row["aa_ids"] = seq_to_aa_ids(row["sequence"])
        return row


def _build_atom14_target(parsed, candidate, *, heavy_coordinates=None):
    names = atom14_names(candidate.sequence, candidate)
    coords = torch.zeros((len(candidate.sequence), 14, 3), dtype=torch.float32)
    mask = torch.zeros((len(candidate.sequence), 14), dtype=torch.bool)
    heavy_coordinates = parsed.heavy_atom_coordinates if heavy_coordinates is None else heavy_coordinates
    for residue, residue_names in enumerate(names):
        for slot, atom_name in enumerate(residue_names):
            if atom_name and atom_name in heavy_coordinates[residue]:
                coords[residue, slot] = heavy_coordinates[residue][atom_name]
                mask[residue, slot] = True
    return coords, mask
