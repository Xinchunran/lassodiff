"""Atom14 covalent graph and bond-geometry audit utilities."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .residue_constants_mini import ELEMENTS, SIDECHAIN_PARENT, padded_atom14_names


@dataclass
class CovalentGraph:
    adjacency: torch.Tensor
    bond_type: torch.Tensor
    bonded_12: torch.Tensor
    bonded_13: torch.Tensor
    bonded_14: torch.Tensor
    peptide_edge_count: int
    isopeptide_edge_count: int


def _connect(adjacency, bond_type, names, ri, ai, rj, aj, code):
    left = {name: slot for slot, name in enumerate(names[ri]) if name}
    right = {name: slot for slot, name in enumerate(names[rj]) if name}
    if ai not in left or aj not in right:
        return False
    i, j = ri * 14 + left[ai], rj * 14 + right[aj]
    adjacency[i, j] = adjacency[j, i] = True
    bond_type[i, j] = bond_type[j, i] = code
    return True


def build_atom14_covalent_graph(aa_ids: torch.Tensor, token_mask: torch.Tensor, candidates: list) -> CovalentGraph:
    if aa_ids.ndim != 2 or token_mask.shape != aa_ids.shape:
        raise ValueError("aa_ids/token_mask must be [B,L]")
    B, L = aa_ids.shape
    adjacency = torch.zeros((B, L * 14, L * 14), dtype=torch.bool, device=aa_ids.device)
    bond_type = torch.zeros((B, L * 14, L * 14), dtype=torch.long, device=aa_ids.device)
    peptides = isopeptides = 0
    for batch, candidate in enumerate(candidates):
        names = padded_atom14_names(candidate.sequence, candidate.k)
        for residue, aa in enumerate(candidate.sequence):
            for left, right in (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")):
                _connect(adjacency[batch], bond_type[batch], names, residue, left, residue, right, 1)
            for atom, parent in SIDECHAIN_PARENT[aa].items():
                _connect(adjacency[batch], bond_type[batch], names, residue, parent, residue, atom, 1)
            if residue + 1 < len(candidate.sequence):
                peptides += int(_connect(adjacency[batch], bond_type[batch], names, residue, "C", residue + 1, "N", 2))
        carbon = "CG" if candidate.sequence[candidate.k] == "D" else "CD"
        isopeptides += int(_connect(adjacency[batch], bond_type[batch], names, 0, "N", candidate.k, carbon, 3))
    bonded12 = adjacency
    bonded13 = torch.bmm(adjacency.float(), adjacency.float()).gt(0) & ~bonded12
    bonded14 = torch.bmm(bonded13.float(), adjacency.float()).gt(0) & ~bonded12 & ~bonded13
    eye = torch.eye(L * 14, dtype=torch.bool, device=aa_ids.device)[None]
    bonded13 &= ~eye; bonded14 &= ~eye
    return CovalentGraph(adjacency, bond_type, bonded12, bonded13, bonded14, peptides, isopeptides)


@dataclass
class BondGeometry:
    length_error: torch.Tensor
    atom_names: tuple[str, ...] = ()


def atom_names_for(aa: str, formed_acceptor: bool = False) -> tuple[str, ...]:
    return padded_atom14_names(aa, 0 if formed_acceptor and aa in "DE" else None)[0]


def atom14_bond_geometry(coordinates: torch.Tensor, mask: torch.Tensor, aa_ids: torch.Tensor, candidates: list) -> BondGeometry:
    if coordinates.ndim != 4:
        raise ValueError("coordinates must be [B,L,14,3]")
    errors = []
    expected = {("N", "CA"): 1.458, ("CA", "C"): 1.525, ("C", "O"): 1.231, ("CA", "CB"): 1.522,
                ("C", "N"): 1.329}
    for batch, candidate in enumerate(candidates):
        names = padded_atom14_names(candidate.sequence, candidate.k)
        for residue, aa in enumerate(candidate.sequence):
            lookup = {name: slot for slot, name in enumerate(names[residue]) if name}
            for left, right in (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB")):
                if left in lookup and right in lookup and mask[batch, residue, lookup[left]] and mask[batch, residue, lookup[right]]:
                    errors.append((coordinates[batch, residue, lookup[left]] - coordinates[batch, residue, lookup[right]]).norm() - expected[(left, right)])
            for atom, parent in SIDECHAIN_PARENT[aa].items():
                if atom in lookup and parent in lookup and mask[batch, residue, lookup[atom]] and mask[batch, residue, lookup[parent]]:
                    expected_side = 1.24 if residue == candidate.k and aa in "DE" and atom in {"OD1", "OE1"} else 1.52
                    errors.append((coordinates[batch, residue, lookup[atom]] - coordinates[batch, residue, lookup[parent]]).norm() - expected_side)
            if residue + 1 < len(candidate.sequence):
                if mask[batch, residue, lookup["C"]] and mask[batch, residue + 1, {n: s for s, n in enumerate(names[residue + 1]) if n}["N"]]:
                    errors.append((coordinates[batch, residue, lookup["C"]] - coordinates[batch, residue + 1, {n: s for s, n in enumerate(names[residue + 1]) if n}["N"]]).norm() - 1.329)
    values = torch.stack(errors) if errors else coordinates.new_zeros((1,))
    return BondGeometry(values.abs().max())


atom14_bond_geometry.atom_names_for = atom_names_for
