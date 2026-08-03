"""Candidate-grouped Mini dataset and fixed-conformer collate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


def grouped_target_mapping_sha256(examples: Iterable[dict[str, Any]]) -> str:
    import hashlib
    import json
    mapping = []
    for row in examples:
        mapping.append({
            "record_id": row["record_id"], "sequence": row["sequence"],
            "k": int(row["k"]), "p": int(row["p"]),
            "loop_size": int(row.get("loop_size", int(row["k"]) + 1)),
            "conformer_count": int(torch.as_tensor(row["conformer_mask"]).sum())
            if "conformer_mask" in row else int(row.get("conformer_count", 0)),
            "ranks": list(row.get("conformer_ranks", ())),
            "sources": list(row.get("conformer_sources", ())),
            "decoder_fit_cache_keys": list(row.get("decoder_fit_cache_keys", ())),
        })
    return hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class GroupedMiniExample:
    record_id: str
    sequence: str
    k: int
    p: int
    core_targets: torch.Tensor
    core_target_masks: torch.Tensor
    atom14_targets: torch.Tensor
    atom14_target_masks: torch.Tensor
    backbone_torsions: torch.Tensor
    backbone_torsion_masks: torch.Tensor
    chi_targets: torch.Tensor
    chi_masks: torch.Tensor
    conformer_mask: torch.Tensor


_TARGET_KEYS = (
    ("core", "core_targets", (7, 3), torch.float32),
    ("core_mask", "core_target_masks", (7,), torch.bool),
    ("atom14", "atom14_targets", (14, 3), torch.float32),
    ("atom14_mask", "atom14_target_masks", (14,), torch.bool),
    ("backbone_torsions", "backbone_torsions", (3,), torch.float32),
    ("backbone_torsion_mask", "backbone_torsion_masks", (3,), torch.bool),
    ("chi", "chi_targets", (4,), torch.float32),
    ("chi_mask", "chi_masks", (4,), torch.bool),
)


def group_candidate_examples(rows: Iterable[dict[str, Any]], max_conformers: int = 3) -> list[dict[str, Any]]:
    if max_conformers < 1:
        raise ValueError("max_conformers must be positive")
    groups: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["record_id"]), str(row["sequence"]), int(row["k"]), int(row["p"]))
        groups.setdefault(key, []).append(row)
    output = []
    for key in sorted(groups):
        record_id, sequence, k, p = key
        members = sorted(groups[key], key=lambda row: (int(row.get("rank", 0)), str(row.get("target", ""))))[:max_conformers]
        length = len(sequence)
        result: dict[str, Any] = {"record_id": record_id, "sequence": sequence, "k": k, "p": p,
                                  "loop_size": int(members[0].get("loop_size", k + 1))}
        result["conformer_ranks"] = tuple(int(x.get("rank", 0)) for x in members)
        result["conformer_sources"] = tuple(str(x.get("target", "")) for x in members)
        result["canonical_roots"] = tuple(x.get("canonical_root") for x in members)
        result["decoder_fit_cache_keys"] = tuple(str(x.get("decoder_fit_cache_key", "")) for x in members)
        result["decoder_fit_metrics"] = tuple({
            "ca_rmsd": x.get("decoder_fit_ca_rmsd"),
            "lddt": x.get("decoder_fit_lddt"),
            "raw_target_strict_valid": x.get("raw_target_strict_valid"),
        } for x in members)
        for source, destination, tail, dtype in _TARGET_KEYS:
            shape = (max_conformers, length, *tail)
            result[destination] = torch.zeros(shape, dtype=dtype)
        result["conformer_mask"] = torch.zeros((max_conformers,), dtype=torch.bool)
        has_raw_atom14 = all("raw_atom14" in member and "raw_atom14_mask" in member for member in members)
        if has_raw_atom14:
            result["raw_atom14_targets"] = torch.zeros((max_conformers, length, 14, 3), dtype=torch.float32)
            result["raw_atom14_target_masks"] = torch.zeros((max_conformers, length, 14), dtype=torch.bool)
        for index, member in enumerate(members):
            result["conformer_mask"][index] = True
            for source, destination, _tail, _dtype in _TARGET_KEYS:
                if source not in member:
                    raise ValueError(f"grouped target row is missing {source}")
                value = torch.as_tensor(member[source])
                expected = result[destination][index].shape
                if value.shape != expected:
                    raise ValueError(f"target {source} has shape {tuple(value.shape)}, expected {tuple(expected)}")
                result[destination][index] = value.to(dtype=result[destination].dtype)
            if has_raw_atom14:
                result["raw_atom14_targets"][index] = torch.as_tensor(member["raw_atom14"], dtype=torch.float32)
                result["raw_atom14_target_masks"][index] = torch.as_tensor(member["raw_atom14_mask"], dtype=torch.bool)
        output.append(result)
    return output


def collate_grouped_mini(items: list[dict[str, Any] | GroupedMiniExample]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty grouped Mini batch")
    rows = [item if isinstance(item, dict) else item.__dict__ for item in items]
    batch_size = len(rows)
    max_conformers = rows[0]["conformer_mask"].shape[0]
    max_length = max(len(row["sequence"]) for row in rows)
    result: dict[str, Any] = {
        "record_ids": [row["record_id"] for row in rows],
        "sequences": [row["sequence"] for row in rows],
        "k": torch.zeros((batch_size,), dtype=torch.long),
        "p": torch.zeros((batch_size,), dtype=torch.long),
        "token_mask": torch.zeros((batch_size, max_length), dtype=torch.bool),
        "loop_size": torch.zeros((batch_size,), dtype=torch.long),
    }
    for _source, destination, tail, dtype in _TARGET_KEYS:
        result[destination] = torch.zeros((batch_size, max_conformers, max_length, *tail), dtype=dtype)
    result["conformer_mask"] = torch.zeros((batch_size, max_conformers), dtype=torch.bool)
    result["conformer_ranks"] = [tuple(row.get("conformer_ranks", ())) for row in rows]
    result["conformer_sources"] = [tuple(row.get("conformer_sources", ())) for row in rows]
    result["canonical_roots"] = [tuple(row.get("canonical_roots", ())) for row in rows]
    result["decoder_fit_cache_keys"] = [tuple(row.get("decoder_fit_cache_keys", ())) for row in rows]
    result["decoder_fit_metrics"] = [tuple(row.get("decoder_fit_metrics", ())) for row in rows]
    result["aa_ids"] = torch.full((batch_size, max_length), 20, dtype=torch.long)
    if all("raw_atom14_targets" in row for row in rows):
        result["raw_atom14_targets"] = torch.zeros((batch_size, max_conformers, max_length, 14, 3), dtype=torch.float32)
        result["raw_atom14_target_masks"] = torch.zeros((batch_size, max_conformers, max_length, 14), dtype=torch.bool)
    for batch_index, row in enumerate(rows):
        length = len(row["sequence"])
        result["k"][batch_index] = int(row["k"])
        result["p"][batch_index] = int(row["p"])
        result["loop_size"][batch_index] = int(row.get("loop_size", int(row["k"]) + 1))
        result["token_mask"][batch_index, :length] = True
        if "aa_ids" in row:
            result["aa_ids"][batch_index, :length] = torch.as_tensor(row["aa_ids"], dtype=torch.long)
        else:
            alphabet = "ACDEFGHIKLMNPQRSTVWY"
            result["aa_ids"][batch_index, :length] = torch.tensor([alphabet.index(aa) for aa in row["sequence"]])
        result["conformer_mask"][batch_index] = row["conformer_mask"]
        for _source, destination, _tail, _dtype in _TARGET_KEYS:
            result[destination][batch_index, :, :length] = row[destination]
        if "raw_atom14_targets" in result:
            result["raw_atom14_targets"][batch_index, :, :length] = row["raw_atom14_targets"]
            result["raw_atom14_target_masks"][batch_index, :, :length] = row["raw_atom14_target_masks"]
    return result


class GroupedMiniDataset(Dataset):
    def __init__(self, rows: Iterable[dict[str, Any]], max_conformers: int = 3):
        self.examples = group_candidate_examples(rows, max_conformers=max_conformers)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
