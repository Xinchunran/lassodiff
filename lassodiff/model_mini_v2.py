"""Residue-level torsion diffusion model for mini_dev."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from .backbone_kinematics import build_core_from_torsions
from .conditioning_mini_v2 import MiniConditioning
from .dynamic_geometry_mini import compute_dynamic_pair_geometry
from .torsion_state import TorsionState, TorsionVelocity


def _time_features(time: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(torch.linspace(0, math.log(1000), half, device=time.device, dtype=time.dtype))
    value = time[:, None] * freq[None] * (2 * math.pi)
    out = torch.cat((torch.sin(value), torch.cos(value)), -1)
    return out if out.shape[-1] == dim else torch.cat((out, time[:, None]), -1)


@dataclass
class MiniTorsionOutput:
    velocity: TorsionVelocity
    dynamic_geometry_calls: int


class _Block(nn.Module):
    def __init__(self, single_dim: int, pair_dim: int, hidden_dim: int):
        super().__init__()
        self.pair = nn.Linear(pair_dim, single_dim)
        self.update = nn.Sequential(nn.LayerNorm(single_dim), nn.Linear(single_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, single_dim))

    def forward(self, single, pair, mask):
        aggregate = self.pair(pair).masked_fill(~mask[..., None], 0).sum(2) / mask.sum(2, keepdim=True).clamp_min(1)
        node_mask = mask.any(dim=2)
        return (single + self.update(single + aggregate)) * node_mask[..., None]


class MiniTorsionDiffusion(nn.Module):
    architecture_id = "lassodiff_mini_torsion_v2"
    schema_version = 2

    def __init__(self, single_dim: int = 256, pair_dim: int = 128, blocks: int = 8,
                 hidden_dim: int = 256, heads: int = 8, dropout: float = .05):
        super().__init__()
        self.single_dim, self.pair_dim = single_dim, pair_dim
        self.blocks = nn.ModuleList(_Block(single_dim, pair_dim, hidden_dim) for _ in range(blocks))
        self.time_projection = nn.Linear(single_dim, single_dim)
        self.geometry_projection = nn.Linear(16, single_dim)
        self.backbone_head = nn.Sequential(nn.LayerNorm(single_dim), nn.Linear(single_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))
        self.chi_head = nn.Sequential(nn.LayerNorm(single_dim), nn.Linear(single_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))

    def forward(self, *, state_t: TorsionState, time: torch.Tensor, conditioning: MiniConditioning,
                token_mask: torch.Tensor, candidates: list):
        if state_t.backbone.ndim != 4:
            raise ValueError("state_t.backbone must be [B,Ns,L,3]")
        B, Ns, L, _ = state_t.backbone.shape
        single = conditioning.single
        if single.shape[:2] != (B, L):
            raise ValueError("conditioning and torsion state shapes differ")
        pair = conditioning.pair
        # Dynamic geometry is deliberately recomputed inside every block.
        dynamic_calls = 0
        torsions = state_t.backbone[:, 0]
        coords = torch.stack([build_core_from_torsions(candidates[b].sequence,
                                                        torsions[b, :len(candidates[b].sequence), 0],
                                                        torsions[b, :len(candidates[b].sequence), 1],
                                                        torsions[b, :len(candidates[b].sequence), 2]) for b in range(B)])
        for block in self.blocks:
            geometry = compute_dynamic_pair_geometry(coords, token_mask)
            dynamic_calls += 1
            # Use invariant scalar geometry channels in the residue update.
            geom_pair = geometry[..., :pair.shape[-1]] if geometry.shape[-1] >= pair.shape[-1] else torch.nn.functional.pad(geometry, (0, pair.shape[-1] - geometry.shape[-1]))
            single = block(single + self.geometry_projection(geometry.mean(2)), pair + geom_pair, conditioning.pair_mask)
        time_feature = self.time_projection(_time_features(time.to(single.dtype), self.single_dim))[:, None]
        single = single + time_feature
        velocity = TorsionVelocity(
            self.backbone_head(single)[:, None].expand(B, Ns, L, 3) * state_t.backbone_mask,
            self.chi_head(single)[:, None].expand(B, Ns, L, 4) * state_t.acceptor_chi_mask,
        )
        return MiniTorsionOutput(velocity, dynamic_calls)
