"""Atom14 masks, rotamer/chi head and deterministic heavy-atom construction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .atom_schema_lasso import (
    ATOM_C, ATOM_CA, ATOM_CB, ATOM_CISO, ATOM_N, ATOM_O, ATOM_OISO,
    CandidateCondition,
)


ATOM14_NAMES = {
    "A": ("N", "CA", "C", "O", "CB"),
    "R": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "N": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    "D": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "C": ("N", "CA", "C", "O", "CB", "SG"),
    "Q": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    "E": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    "G": ("N", "CA", "C", "O"),
    "H": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "I": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "L": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    "K": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    "M": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    "F": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "P": ("N", "CA", "C", "O", "CB", "CG", "CD"),
    "S": ("N", "CA", "C", "O", "CB", "OG"),
    "T": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "W": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "Y": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "V": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
}

SYMMETRIC_ATOM_PAIRS = {
    "D": (("OD1", "OD2"),), "E": (("OE1", "OE2"),),
    "F": (("CD1", "CD2"), ("CE1", "CE2")),
    "Y": (("CD1", "CD2"), ("CE1", "CE2")),
    "R": (("NH1", "NH2"),),
}


def atom14_names(sequence: str, candidate: CandidateCondition | None = None) -> tuple[tuple[str, ...], ...]:
    rows = []
    for index, aa in enumerate(sequence):
        names = list(ATOM14_NAMES[aa])
        if candidate is not None and index == candidate.k:
            removed = "OD2" if aa == "D" else "OE2"
            names = [name for name in names if name != removed]
        rows.append(tuple(names + [""] * (14 - len(names))))
    return tuple(rows)


def atom14_mask(sequence: str, candidate: CandidateCondition | None = None) -> torch.Tensor:
    return torch.tensor([[bool(name) for name in row] for row in atom14_names(sequence, candidate)], dtype=torch.bool)


@dataclass(frozen=True)
class SidechainPrediction:
    rotamer_logits: torch.Tensor
    chi_sin_cos: torch.Tensor


class RotamerChiHead(nn.Module):
    def __init__(self, residue_dim: int, hidden_dim: int = 128, rotamer_classes: int = 3):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(residue_dim), nn.Linear(residue_dim, hidden_dim), nn.SiLU())
        self.rotamer = nn.Linear(hidden_dim, rotamer_classes)
        self.chi = nn.Linear(hidden_dim, 4 * 2)

    def forward(self, residue_representation: torch.Tensor) -> SidechainPrediction:
        hidden = self.trunk(residue_representation)
        chi = self.chi(hidden).reshape(*hidden.shape[:-1], 4, 2)
        chi = F.normalize(chi, dim=-1, eps=1e-8)
        return SidechainPrediction(self.rotamer(hidden), chi)


def _local_frame(core_row: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ca = core_row[ATOM_CA]
    x = core_row[ATOM_CB] - ca
    x = x / x.norm().clamp_min(1e-8)
    y = core_row[ATOM_C] - ca
    y = y - torch.dot(y, x) * x
    y = y / y.norm().clamp_min(1e-8)
    z = torch.linalg.cross(x, y, dim=-1)
    return x, y, z / z.norm().clamp_min(1e-8)


def build_atom14(
    core_coordinates: torch.Tensor,
    candidate: CandidateCondition,
    chi_sin_cos: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct complete heavy atoms; formed acceptor keeps Stage-1 CISO/OISO."""
    length = len(candidate.sequence)
    if core_coordinates.shape != (length, 7, 3):
        raise ValueError("core coordinates must have shape [L,7,3]")
    names = atom14_names(candidate.sequence, candidate)
    mask = atom14_mask(candidate.sequence, candidate).to(core_coordinates.device)
    output = torch.zeros((length, 14, 3), dtype=core_coordinates.dtype, device=core_coordinates.device)
    for residue, aa in enumerate(candidate.sequence):
        lookup = {name: slot for slot, name in enumerate(names[residue]) if name}
        for atom_name, core_slot in (("N", ATOM_N), ("CA", ATOM_CA), ("C", ATOM_C), ("O", ATOM_O), ("CB", ATOM_CB)):
            if atom_name in lookup:
                output[residue, lookup[atom_name]] = core_coordinates[residue, core_slot]
        if aa == "G":
            continue
        x, y, z = _local_frame(core_coordinates[residue])
        cb = core_coordinates[residue, ATOM_CB]
        side_names = [name for name in names[residue][5:] if name]
        for offset, atom_name in enumerate(side_names):
            chi_index = min(offset, 3)
            if chi_sin_cos is None:
                sine = core_coordinates.new_tensor(0.0)
                cosine = core_coordinates.new_tensor(-1.0 if offset % 2 else 1.0)
            else:
                sine = chi_sin_cos[residue, chi_index, 0]
                cosine = chi_sin_cos[residue, chi_index, 1]
            radial = 1.50 * (offset + 1)
            branch = (offset // 2 + 1) * .35
            output[residue, lookup[atom_name]] = cb + radial * x + branch * (cosine * y + sine * z)
        if residue == candidate.k:
            carbon = "CG" if aa == "D" else "CD"
            oxygen = "OD1" if aa == "D" else "OE1"
            output[residue, lookup[carbon]] = core_coordinates[residue, ATOM_CISO]
            output[residue, lookup[oxygen]] = core_coordinates[residue, ATOM_OISO]
            if aa == "E" and "CG" in lookup:
                # The strict checker uses the real predecessor S=CG for Glu.
                cb_to_ciso = core_coordinates[residue, ATOM_CISO] - cb
                output[residue, lookup["CG"]] = cb + .5 * cb_to_ciso
    return output, mask


def symmetry_aware_coordinate_loss(
    predicted: torch.Tensor, target: torch.Tensor, sequence: str, mask: torch.Tensor,
    candidate: CandidateCondition,
) -> torch.Tensor:
    """Minimum-permutation coordinate loss, excluding formed acceptor symmetry."""
    names = atom14_names(sequence, candidate)
    losses = []
    for residue, aa in enumerate(sequence):
        valid = mask[residue]
        base = (predicted[residue, valid] - target[residue, valid]).square().sum(-1).mean()
        alternatives = [base]
        if residue != candidate.k:
            lookup = {name: slot for slot, name in enumerate(names[residue]) if name}
            for left, right in SYMMETRIC_ATOM_PAIRS.get(aa, ()):
                if left in lookup and right in lookup:
                    swapped = target[residue].clone()
                    swapped[[lookup[left], lookup[right]]] = swapped[[lookup[right], lookup[left]]]
                    alternatives.append((predicted[residue, valid] - swapped[valid]).square().sum(-1).mean())
        losses.append(torch.stack(alternatives).min())
    return torch.stack(losses).mean()
