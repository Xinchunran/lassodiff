"""Chemical schemas and fail-closed candidate contracts for LassoDiff Mini."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import torch


MINI_SCHEMA_VERSION = 1
MINI_TORSION_SCHEMA_VERSION = 2
LEGACY_MINI_ARCHITECTURE_ID = "lassodiff_mini_core7_legacy"
MINI_TORSION_ARCHITECTURE_ID = "lassodiff_mini_torsion_v2"
CORE_ATOM_NAMES = ("N", "CA", "C", "O", "CB", "CISO", "OISO")


class CoreAtom(IntEnum):
    N = 0
    CA = 1
    C = 2
    O = 3
    CB = 4
    CISO = 5
    OISO = 6


ATOM_N = int(CoreAtom.N)
ATOM_CA = int(CoreAtom.CA)
ATOM_C = int(CoreAtom.C)
ATOM_O = int(CoreAtom.O)
ATOM_CB = int(CoreAtom.CB)
ATOM_CISO = int(CoreAtom.CISO)
ATOM_OISO = int(CoreAtom.OISO)

CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class CovalentEdge:
    left_residue: int
    left_atom: str
    right_residue: int
    right_atom: str
    bond_type: str = "single"


@dataclass(frozen=True)
class CandidateCondition:
    """One sequence/topology hypothesis; invalid hypotheses are never clamped."""

    sequence: str
    acceptor_index: int
    plug_index: int
    min_post_plug_tail: int = 1

    def __post_init__(self) -> None:
        sequence = "".join(self.sequence.upper().split())
        object.__setattr__(self, "sequence", sequence)
        if not sequence or any(aa not in CANONICAL_AA for aa in sequence):
            raise ValueError("candidate sequence must contain canonical amino acids")
        length = len(sequence)
        k, p = int(self.acceptor_index), int(self.plug_index)
        if not 0 <= k < length:
            raise ValueError("candidate acceptor index is outside the sequence")
        if sequence[k] not in "DE":
            raise ValueError("candidate acceptor must be Asp or Glu")
        if not k < p < length:
            raise ValueError("candidate requires acceptor_index < plug_index < length")
        if length - p - 1 < int(self.min_post_plug_tail):
            raise ValueError("candidate post-plug tail is too short")

    @property
    def k(self) -> int:
        return self.acceptor_index

    @property
    def p(self) -> int:
        return self.plug_index

    @property
    def acceptor_type(self) -> str:
        return "ASP_ISO" if self.sequence[self.k] == "D" else "GLU_ISO"

    @property
    def ring_mask(self) -> torch.Tensor:
        index = torch.arange(len(self.sequence))
        return index <= self.k

    @property
    def tail_mask(self) -> torch.Tensor:
        index = torch.arange(len(self.sequence))
        return index > self.p

    @property
    def core_atom_mask(self) -> torch.Tensor:
        return core_atom_mask(self.sequence, self.k)

    @property
    def covalent_edges(self) -> tuple[CovalentEdge, ...]:
        edges: list[CovalentEdge] = []
        for residue, aa in enumerate(self.sequence):
            edges.extend((
                CovalentEdge(residue, "N", residue, "CA"),
                CovalentEdge(residue, "CA", residue, "C"),
                CovalentEdge(residue, "C", residue, "O", "double"),
            ))
            if aa != "G":
                edges.append(CovalentEdge(residue, "CA", residue, "CB"))
            if residue + 1 < len(self.sequence):
                edges.append(CovalentEdge(residue, "C", residue + 1, "N", "peptide"))
        if self.sequence[self.k] == "D":
            edges.append(CovalentEdge(self.k, "CB", self.k, "CISO"))
        # GLU's direct predecessor is CG, which is intentionally not in core-7.
        edges.append(CovalentEdge(self.k, "CISO", self.k, "OISO", "double"))
        edges.append(CovalentEdge(0, "N", self.k, "CISO", "isopeptide"))
        return tuple(edges)


def core_atom_mask(sequence: str, acceptor_index: int) -> torch.Tensor:
    sequence = "".join(sequence.upper().split())
    if not sequence or not 0 <= acceptor_index < len(sequence):
        raise ValueError("cannot build atom mask for invalid sequence/acceptor")
    if sequence[acceptor_index] not in "DE":
        raise ValueError("core-7 acceptor must be Asp or Glu")
    mask = torch.zeros((len(sequence), len(CORE_ATOM_NAMES)), dtype=torch.bool)
    mask[:, :4] = True
    for index, aa in enumerate(sequence):
        mask[index, ATOM_CB] = aa != "G"
    mask[acceptor_index, ATOM_CISO:ATOM_OISO + 1] = True
    return mask


def candidate_conditions(
    sequence: str, acceptor_index: int, plug_indices: Iterable[int], *, min_post_plug_tail: int = 1,
) -> tuple[CandidateCondition, ...]:
    """Expand candidate-specific examples instead of sharing one k/p across targets."""
    return tuple(
        CandidateCondition(sequence, acceptor_index, int(plug), min_post_plug_tail)
        for plug in plug_indices
    )
