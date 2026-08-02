"""Small sequence/candidate-conditioned equivariant core-7 diffusion model."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from .topology_adapter import CandidateBatch


ARCHITECTURE_ID_MINI = "lassodiff_mini_core7"
SCHEMA_VERSION_MINI = 1


def _time_features(t: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequency = torch.exp(torch.linspace(0, math.log(1000.0), half, device=t.device, dtype=t.dtype))
    phase = t[:, None] * frequency[None] * (2 * math.pi)
    features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
    return features if dimension % 2 == 0 else torch.cat((features, t[:, None]), dim=-1)


class _CoreEGNNBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.coordinate = nn.Linear(hidden_dim, 1)
        self.update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, h, x, mask):
        relative = x[:, :, None] - x[:, None, :]
        distance2 = relative.square().sum(-1, keepdim=True).clamp_max(400.0)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair_mask &= ~torch.eye(x.shape[1], dtype=torch.bool, device=x.device)[None]
        message = self.message(torch.cat((
            h[:, :, None].expand(-1, -1, h.shape[1], -1),
            h[:, None, :].expand(-1, h.shape[1], -1, -1), distance2,
        ), dim=-1)) * pair_mask[..., None]
        count = pair_mask.sum(-1, keepdim=True).clamp_min(1).to(x.dtype)
        delta = (relative * self.coordinate(message)).sum(2) / count
        aggregate = message.sum(2) / count
        return (h + self.update(torch.cat((h, aggregate), -1))) * mask[..., None], x + delta * mask[..., None]


@dataclass(frozen=True)
class MiniCoreOutput:
    velocity: torch.Tensor
    candidate_mask: torch.Tensor
    dynamic_geometry_calls: int
    residue_representation: torch.Tensor


class MiniCoreDiffusion(nn.Module):
    architecture_id = ARCHITECTURE_ID_MINI
    schema_version = SCHEMA_VERSION_MINI

    def __init__(self, hidden_dim: int = 128, blocks: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aa_embedding = nn.Embedding(21, hidden_dim)
        self.atom_embedding = nn.Embedding(7, hidden_dim)
        self.role_projection = nn.Linear(6, hidden_dim)
        self.time_projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList(_CoreEGNNBlock(hidden_dim) for _ in range(blocks))

    def forward(self, aa_ids, token_mask, candidates: CandidateBatch, x_t, t, atom_mask):
        B, M, L, A, coordinate_dim = x_t.shape
        if coordinate_dim != 3 or A != 7 or aa_ids.shape != (B, L) or token_mask.shape != (B, L):
            raise ValueError("Mini core model input shape mismatch")
        if atom_mask.shape != (B, M, L, A) or t.shape != (B,):
            raise ValueError("Mini atom mask/time shape mismatch")
        candidates.validate(L)
        index = torch.arange(L, device=x_t.device)[None, None].expand(B, M, L)
        k, p, acceptor = candidates.k[..., None], candidates.p[..., None], candidates.acceptor_index[..., None]
        roles = torch.stack((
            (index <= k).to(x_t.dtype), ((index > k) & (index <= p)).to(x_t.dtype),
            (index > p).to(x_t.dtype), (index == p).to(x_t.dtype),
            (index == acceptor).to(x_t.dtype), (index == 0).to(x_t.dtype),
        ), -1)
        residue = self.aa_embedding(aa_ids)[:, None].expand(B, M, L, -1) + self.role_projection(roles)
        atom_ids = torch.arange(A, device=x_t.device)[None, None, None].expand(B, M, L, A)
        hidden = residue[:, :, :, None] + self.atom_embedding(atom_ids)
        hidden = hidden + self.time_projection(_time_features(t, self.hidden_dim))[:, None, None, None]
        flat_mask = (atom_mask & token_mask[:, None, :, None] & candidates.candidate_mask[:, :, None, None]).reshape(B * M, L * A)
        hidden = hidden.reshape(B * M, L * A, self.hidden_dim) * flat_mask[..., None]
        initial = x_t.reshape(B * M, L * A, 3)
        coordinates = initial
        for block in self.blocks:
            hidden, coordinates = block(hidden, coordinates, flat_mask)
        velocity = (coordinates - initial).reshape(B, M, L, A, 3) * atom_mask[..., None]
        residue_hidden = hidden.reshape(B, M, L, A, self.hidden_dim)
        atom_weight = atom_mask[..., None].to(residue_hidden.dtype)
        residue_hidden = (residue_hidden * atom_weight).sum(-2) / atom_weight.sum(-2).clamp_min(1)
        return MiniCoreOutput(velocity, candidates.candidate_mask, len(self.blocks), residue_hidden)
