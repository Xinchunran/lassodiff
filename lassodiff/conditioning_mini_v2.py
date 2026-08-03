"""Sequence and candidate conditioning for the torsion diffusion graph."""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from .atom_schema_lasso import CandidateCondition
from .esm_encoder_mini import FrozenESMResidueEncoder


class MiniConditioning(NamedTuple):
    single: torch.Tensor
    pair: torch.Tensor
    pair_mask: torch.Tensor


class MiniSequenceConditioner(nn.Module):
    def __init__(self, residue_encoder: nn.Module | None = None, residue_encoder_dim: int = 640,
                 single_dim: int = 256, pair_dim: int = 128):
        super().__init__()
        self.residue_encoder = residue_encoder or FrozenESMResidueEncoder(output_dim=residue_encoder_dim)
        for parameter in self.residue_encoder.parameters():
            parameter.requires_grad_(False)
        self.residue_encoder.eval()
        self.single_dim = single_dim
        self.pair_dim = pair_dim
        self.aa_embedding = nn.Embedding(21, 64)
        self.esm_projection = nn.Linear(residue_encoder_dim, single_dim)
        self.esm_gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.single_projection = nn.Sequential(nn.Linear(single_dim + 64 + 8, single_dim), nn.SiLU(), nn.Linear(single_dim, single_dim))
        self.pair_projection = nn.Sequential(nn.Linear(7, pair_dim), nn.SiLU(), nn.Linear(pair_dim, pair_dim))

    def train(self, mode: bool = True):
        super().train(mode)
        self.residue_encoder.eval()
        return self

    def forward(self, *, sequences: list[str], aa_ids: torch.Tensor, token_mask: torch.Tensor,
                k: torch.Tensor, p: torch.Tensor) -> MiniConditioning:
        if aa_ids.ndim != 2 or token_mask.shape != aa_ids.shape:
            raise ValueError("aa_ids/token_mask must have shape [B,L]")
        esm = self.residue_encoder(sequences, token_mask).detach()
        B, L = aa_ids.shape
        index = torch.arange(L, device=aa_ids.device, dtype=esm.dtype)[None, :].expand(B, L)
        kk, pp = k.to(index.dtype).reshape(B, 1), p.to(index.dtype).reshape(B, 1)
        roles = torch.stack(((index <= kk).to(esm.dtype), ((index > kk) & (index <= pp)).to(esm.dtype),
                             (index > pp).to(esm.dtype), (index == kk).to(esm.dtype),
                             (index == pp).to(esm.dtype), (index == 0).to(esm.dtype),
                             (index - kk).abs() / max(L, 1), (index - pp).abs() / max(L, 1)), -1)
        esm = torch.sigmoid(self.esm_gate_logit) * self.esm_projection(esm)
        single = self.single_projection(torch.cat((esm, self.aa_embedding(aa_ids), roles), -1))
        rel = index[:, :, None] - index[:, None, :]
        pair_static = torch.stack((rel / max(L, 1), rel.abs() / max(L, 1),
                                    (rel == 1).to(esm.dtype), (rel == -1).to(esm.dtype),
                                    ((index[:, :, None] <= kk[:, :, None]) == (index[:, None, :] <= kk[:, None, :])).to(esm.dtype),
                                    ((index[:, :, None] <= pp[:, :, None]) == (index[:, None, :] <= pp[:, None, :])).to(esm.dtype),
                                    ((index[:, :, None] == pp[:, :, None]) | (index[:, None, :] == pp[:, None, :])).to(esm.dtype)), -1)
        pair = self.pair_projection(pair_static)
        pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
        return MiniConditioning(single * token_mask[..., None], pair * pair_mask[..., None], pair_mask)
