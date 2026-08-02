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
from .mini_grouped_dataset import group_candidate_examples


class GroupedMiniPDBDataset(Dataset):
    """One dataset item per (record, sequence, k, p) candidate."""

    def __init__(self, metadata_json: str | Path, structure_root: str | Path,
                 *, record_ids: Iterable[str] | None = None, max_conformers: int = 3):
        rows = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
        root = Path(structure_root)
        allowed = None if record_ids is None else set(map(str, record_ids))
        target_rows: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        for metadata in rows:
            record_id = str(metadata.get("LP_ID") or "")
            if not record_id or (allowed is not None and record_id not in allowed):
                continue
            sequence = "".join(str(metadata.get("Core_Sequence") or "").upper().split())
            try:
                k = int(metadata.get("Ring_Length") or 0) - 1
                if not sequence or sequence[k] not in "DE":
                    raise ValueError("invalid acceptor")
            except (ValueError, IndexError, TypeError) as exc:
                self.rejections.append({"record_id": record_id, "reason": str(exc)})
                continue
            for rank in range(1, 4):
                raw_plug = metadata.get(f"Upper_Plug_{rank}")
                if raw_plug in (None, ""):
                    continue
                try:
                    candidate = CandidateCondition(sequence, k, int(raw_plug) - 1)
                except ValueError as exc:
                    self.rejections.append({"record_id": record_id, "rank": rank, "reason": str(exc)})
                    continue
                parsed = None
                selected = None
                failures = []
                for path in (root / record_id / f"relax{rank}.pdb", root / record_id / f"min{rank}.pdb"):
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
                atom14, atom14_mask = _build_atom14_target(parsed, candidate)
                required14 = torch.tensor([[bool(name) for name in names] for names in atom14_names(sequence, candidate)], dtype=torch.bool)
                if not bool(atom14_mask[required14].all()):
                    self.rejections.append({"record_id": record_id, "rank": rank, "reason": "missing_required_atom14_target"})
                    continue
                # A target with missing chi atoms cannot supervise the generated
                # chi state.  Reject it rather than invent coordinates.
                torsions, torsion_mask = extract_backbone_torsions(parsed.core_coordinates, parsed.core_atom_mask)
                chi = extract_chi_angles(atom14[None, None], atom14_mask[None, None],
                                         seq_to_aa_ids(sequence)[None], [candidate])
                target_rows.append({
                    "record_id": record_id, "sequence": sequence, "k": candidate.k, "p": candidate.p,
                    "rank": rank, "target": str(selected.resolve()),
                    "core": parsed.core_coordinates.float(), "core_mask": parsed.core_atom_mask.bool(),
                    "atom14": atom14, "atom14_mask": atom14_mask,
                    "backbone_torsions": torsions, "backbone_torsion_mask": torsion_mask,
                    "chi": chi.angles[0, 0], "chi_mask": chi.masks[0, 0],
                })
        if not target_rows:
            raise ValueError("Grouped Mini V2 dataset contains no valid targets")
        self.examples = group_candidate_examples(target_rows, max_conformers=max_conformers)
        self.qualified_record_ids = tuple(sorted({x["record_id"] for x in self.examples}))
        self.missing_record_ids = tuple(sorted(allowed - set(self.qualified_record_ids))) if allowed is not None else ()
        mapping = [{"record_id": x["record_id"], "sequence": x["sequence"], "k": int(x["k"]), "p": int(x["p"]),
                    "ranks": list(x.get("conformer_ranks", ())), "sources": list(x.get("conformer_sources", ())) }
                   for x in self.examples]
        self.mapping_sha256 = hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        row = dict(self.examples[index])
        row["aa_ids"] = seq_to_aa_ids(row["sequence"])
        return row


def _build_atom14_target(parsed, candidate):
    names = atom14_names(candidate.sequence, candidate)
    coords = torch.zeros((len(candidate.sequence), 14, 3), dtype=torch.float32)
    mask = torch.zeros((len(candidate.sequence), 14), dtype=torch.bool)
    for residue, residue_names in enumerate(names):
        for slot, atom_name in enumerate(residue_names):
            if atom_name and atom_name in parsed.heavy_atom_coordinates[residue]:
                coords[residue, slot] = parsed.heavy_atom_coordinates[residue][atom_name]
                mask[residue, slot] = True
    return coords, mask
