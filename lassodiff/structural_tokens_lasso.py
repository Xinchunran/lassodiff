from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .topology_adapter import CandidateBatch


ROLE_BACKBONE = 0
ROLE_SIDECHAIN = 1
ROLE_NTERM_REACTIVE = 2
ROLE_ACCEPTOR_CARBOXYL = 3
ROLE_PLUG = 4
ROLE_RING_CONTEXT = 5
N_ROLES = 6


@dataclass
class StructuralTokenState:
    single: torch.Tensor
    pair: torch.Tensor
    token_mask: torch.Tensor
    parent_residue: torch.Tensor
    role_id: torch.Tensor
    atom_map: torch.Tensor


class LassoStructuralTokenExpander(nn.Module):
    def __init__(self, c_s: int, c_z: int):
        super().__init__()
        self.role_single = nn.Parameter(torch.randn(N_ROLES, c_s) * 0.02)
        self.role_pair = nn.Embedding(N_ROLES * N_ROLES, c_z)
        self.same_parent = nn.Parameter(torch.randn(c_z) * 0.02)

    def forward(self, s, z, residue_mask, candidates: CandidateBatch) -> StructuralTokenState:
        B, M, L, C_s = s.shape
        T = 2 * L + 4
        parent = torch.zeros((B, M, T), dtype=torch.long, device=s.device)
        roles = torch.zeros_like(parent)
        token_mask = torch.zeros((B, M, T), dtype=torch.bool, device=s.device)
        index = torch.arange(L, device=s.device)
        parent[:, :, :L] = index
        parent[:, :, L:2 * L] = index
        roles[:, :, :L] = ROLE_BACKBONE
        roles[:, :, L:2 * L] = ROLE_SIDECHAIN
        parent[:, :, 2 * L] = 0
        parent[:, :, 2 * L + 1] = candidates.acceptor_index
        parent[:, :, 2 * L + 2] = candidates.p
        parent[:, :, 2 * L + 3] = candidates.k
        roles[:, :, 2 * L:] = torch.tensor(
            [ROLE_NTERM_REACTIVE, ROLE_ACCEPTOR_CARBOXYL, ROLE_PLUG, ROLE_RING_CONTEXT],
            device=s.device,
        )
        candidate_valid = candidates.candidate_mask[:, :, None]
        token_mask[:, :, :L] = residue_mask[:, None] & candidate_valid
        token_mask[:, :, L:2 * L] = residue_mask[:, None] & candidate_valid
        token_mask[:, :, 2 * L:] = candidate_valid.expand(B, M, 4)

        single_index = parent[..., None].expand(B, M, T, C_s)
        token_single = torch.gather(s, 2, single_index) + self.role_single[roles]
        C_z = z.shape[-1]
        flat_z = z.reshape(B, M, L * L, C_z)
        pair_index = (parent[:, :, :, None] * L + parent[:, :, None, :]).reshape(B, M, T * T)
        token_pair = torch.gather(
            flat_z, 2, pair_index[..., None].expand(B, M, T * T, C_z)
        ).reshape(B, M, T, T, C_z)
        role_pair_index = roles[:, :, :, None] * N_ROLES + roles[:, :, None, :]
        token_pair = token_pair + self.role_pair(role_pair_index)
        token_pair = token_pair + (parent[:, :, :, None] == parent[:, :, None, :])[..., None] * self.same_parent
        structural_pair_mask = token_mask[:, :, :, None] & token_mask[:, :, None, :]
        token_single = token_single * token_mask[..., None].to(token_single.dtype)
        token_pair = token_pair * structural_pair_mask[..., None].to(token_pair.dtype)
        return StructuralTokenState(token_single, token_pair, token_mask, parent, roles, parent.clone())

    def collapse_to_residue(self, state: StructuralTokenState, length: int):
        B, M, T, C_s = state.single.shape
        single = state.single.new_zeros((B, M, length, C_s))
        single_count = state.single.new_zeros((B, M, length, 1))
        weight = state.token_mask[..., None].to(state.single.dtype)
        single.scatter_add_(2, state.parent_residue[..., None].expand(B, M, T, C_s), state.single * weight)
        single_count.scatter_add_(2, state.parent_residue[..., None], weight)
        single = single / single_count.clamp(min=1.0)

        C_z = state.pair.shape[-1]
        pair = state.pair.new_zeros((B, M, length * length, C_z))
        pair_count = state.pair.new_zeros((B, M, length * length, 1))
        flat_index = (state.parent_residue[:, :, :, None] * length + state.parent_residue[:, :, None, :]).reshape(B, M, T * T)
        pair_weight = (state.token_mask[:, :, :, None] & state.token_mask[:, :, None, :]).reshape(B, M, T * T, 1).to(state.pair.dtype)
        pair.scatter_add_(2, flat_index[..., None].expand(B, M, T * T, C_z), state.pair.reshape(B, M, T * T, C_z) * pair_weight)
        pair_count.scatter_add_(2, flat_index[..., None], pair_weight)
        pair = (pair / pair_count.clamp(min=1.0)).reshape(B, M, length, length, C_z)
        return single, pair
