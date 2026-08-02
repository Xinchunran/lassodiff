"""Candidate-expanded raw PDB dataset for the Mini chemical schema."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from ..atom_schema_lasso import CandidateCondition
from ..seq_encoder import seq_to_aa_ids
from ..sidechain_builder import atom14_names
from ..structure_processor import process_lasso_structure


class MiniLassoDataset(Dataset):
    """Each item is exactly one (sequence,k,p,rank-matched structure) target."""

    def __init__(self, metadata_json: str | Path, structure_root: str | Path, record_ids=None):
        rows = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
        allowed = None if record_ids is None else set(record_ids)
        self.requested_record_ids = None if allowed is None else tuple(sorted(allowed))
        examples: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        for row in rows:
            record_id = str(row.get("LP_ID") or "")
            if not record_id or (allowed is not None and record_id not in allowed):
                continue
            sequence = "".join(str(row.get("Core_Sequence") or "").upper().split())
            k = int(row.get("Ring_Length") or 0) - 1
            if not sequence or not 0 <= k < len(sequence) or sequence[k] not in "DE":
                continue
            for rank in range(1, 4):
                raw_plug = row.get(f"Upper_Plug_{rank}")
                if raw_plug in (None, ""):
                    continue
                try:
                    candidate = CandidateCondition(sequence, k, int(raw_plug) - 1)
                except ValueError:
                    continue
                root = Path(structure_root) / record_id
                targets = [root / name for name in (f"relax{rank}.pdb", f"min{rank}.pdb") if (root / name).is_file()]
                target, failures = None, []
                for path in targets:
                    try:
                        process_lasso_structure(path, candidate)
                        target = path
                        break
                    except (ValueError, OSError) as exc:
                        failures.append({"target": path.name, "reason": str(exc)})
                if target is not None:
                    examples.append({"record_id": record_id, "rank": rank, "candidate": candidate, "target": target})
                elif targets:
                    rejections.append({"record_id": record_id, "rank": rank, "failures": failures})
        if not examples:
            raise ValueError("Mini dataset contains no candidate-specific examples")
        self.examples = examples
        self.rejections = rejections
        self.qualified_record_ids = tuple(sorted({item["record_id"] for item in examples}))
        self.missing_record_ids = tuple(
            sorted(allowed - set(self.qualified_record_ids)) if allowed is not None else (),
        )
        mapping = [
            {
                "record_id": item["record_id"], "rank": item["rank"],
                "sequence": item["candidate"].sequence, "k": item["candidate"].k,
                "p": item["candidate"].p, "target": str(item["target"].resolve()),
            }
            for item in examples
        ]
        self.mapping_sha256 = hashlib.sha256(
            json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        ).hexdigest()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        candidate = example["candidate"]
        structure = process_lasso_structure(example["target"], candidate)
        names = atom14_names(candidate.sequence, candidate)
        atom14 = torch.zeros((len(candidate.sequence), 14, 3), dtype=torch.float32)
        atom14_mask = torch.zeros((len(candidate.sequence), 14), dtype=torch.bool)
        for residue, row_names in enumerate(names):
            for slot, name in enumerate(row_names):
                if name and name in structure.heavy_atom_coordinates[residue]:
                    atom14[residue, slot] = structure.heavy_atom_coordinates[residue][name]
                    atom14_mask[residue, slot] = True
        return {
            "record_id": example["record_id"], "rank": example["rank"], "candidate": candidate,
            "aa_ids": seq_to_aa_ids(candidate.sequence), "core": structure.core_coordinates,
            "core_mask": structure.core_atom_mask, "atom14": atom14, "atom14_mask": atom14_mask,
        }


def collate_mini(items):
    if not items:
        raise ValueError("cannot collate an empty Mini batch")
    B, L = len(items), max(len(item["candidate"].sequence) for item in items)
    aa_ids = torch.full((B, L), 20, dtype=torch.long)
    token_mask = torch.zeros((B, L), dtype=torch.bool)
    core = torch.zeros((B, 1, L, 7, 3))
    core_mask = torch.zeros((B, 1, L, 7), dtype=torch.bool)
    atom14 = torch.zeros((B, L, 14, 3))
    atom14_mask = torch.zeros((B, L, 14), dtype=torch.bool)
    k = torch.zeros((B, 1), dtype=torch.long)
    p = torch.zeros((B, 1), dtype=torch.long)
    acceptor_type = torch.zeros((B, 1), dtype=torch.long)
    candidates = []
    for b, item in enumerate(items):
        candidate = item["candidate"]
        length = len(candidate.sequence)
        aa_ids[b, :length] = item["aa_ids"]
        token_mask[b, :length] = True
        core[b, 0, :length] = item["core"]
        core_mask[b, 0, :length] = item["core_mask"]
        atom14[b, :length] = item["atom14"]
        atom14_mask[b, :length] = item["atom14_mask"]
        k[b, 0] = candidate.k
        p[b, 0] = candidate.p
        acceptor_type[b, 0] = candidate.sequence[candidate.k] == "E"
        candidates.append(candidate)
    return {
        "aa_ids": aa_ids, "token_mask": token_mask, "core": core, "core_mask": core_mask,
        "atom14": atom14, "atom14_mask": atom14_mask, "k": k, "p": p,
        "acceptor_type": acceptor_type, "conditions": candidates,
    }
