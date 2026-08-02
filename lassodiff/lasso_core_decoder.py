"""Unified differentiable lasso-core decoder for torsion states."""
from __future__ import annotations

import torch

from .atom_schema_lasso import ATOM_CISO, ATOM_OISO, CandidateCondition
from .backbone_kinematics import build_core_batch
from .chi_geometry import build_acceptor_reactive_atoms
from .torsion_state import TorsionState


def decode_lasso_core(state: TorsionState, *, sequences: list[str],
                      candidates: list[CandidateCondition], token_mask: torch.Tensor) -> torch.Tensor:
    """Decode [B,Ns,L,3] torsions to padded [B,Ns,L,7,3] core coordinates."""
    B, Ns, L, _ = state.backbone.shape
    if len(sequences) != B or len(candidates) != B or token_mask.shape != (B, L):
        raise ValueError("state, candidates, sequences and token_mask disagree")
    flat_state = state.backbone.reshape(B * Ns, L, 3)
    flat_mask = token_mask[:, None].expand(B, Ns, L).reshape(B * Ns, L)
    flat_sequences = [sequence for sequence in sequences for _ in range(Ns)]
    core = build_core_batch(flat_sequences, flat_state, flat_mask).reshape(B, Ns, L, 7, 3)
    for b, candidate in enumerate(candidates):
        length = len(candidate.sequence)
        for sample in range(Ns):
            ciso, oiso = build_acceptor_reactive_atoms(
                sequence=candidate.sequence,
                core=core[b, sample, :length],
                acceptor_index=candidate.k,
                chi=state.acceptor_chi[b, sample, candidate.k],
                chi_mask=state.acceptor_chi_mask[b, sample, candidate.k],
                n_terminal_position=core[b, sample, 0, 0],
            )
            core[b, sample, candidate.k, ATOM_CISO] = ciso
            core[b, sample, candidate.k, ATOM_OISO] = oiso
    return core * token_mask[:, None, :, None, None].to(core.dtype)
