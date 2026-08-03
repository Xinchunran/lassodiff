"""Residue-level, sequence-conditioned torsion diffusion for Mini V2."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from .conditioning_mini_v2 import MiniConditioning
from .dynamic_geometry_mini import compute_dynamic_pair_geometry
from .lasso_core_decoder import decode_lasso_core
from .torsion_state import TorsionState, TorsionVelocity
from .torsion_flow import wrap_angle


def _time_features(time: torch.Tensor, dim: int) -> torch.Tensor:
    flat = time.reshape(-1)
    half = max(dim // 2, 1)
    # Rectified-flow time is sampled on [0, 1].  Frequencies growing to 1000
    # make adjacent ODE steps look unrelated and allow point-wise flow loss to
    # decrease while integration leaves the learned path.  Use the standard
    # smooth transformer spectrum (1 -> 1e-4) so the velocity field remains
    # interpolable by Euler/Heun between supervised time points.
    denominator = max(half - 1, 1)
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / denominator
    )
    value = flat[:, None] * freq[None] * (2 * math.pi)
    out = torch.cat((torch.sin(value), torch.cos(value)), -1)
    return out[:, :dim] if out.shape[-1] >= dim else torch.cat((out, flat[:, None]), -1)[:, :dim]


@dataclass
class MiniTorsionOutput:
    velocity: TorsionVelocity
    dynamic_geometry_calls: int


class _Block(nn.Module):
    def __init__(self, single_dim: int, pair_dim: int, hidden_dim: int):
        super().__init__()
        self.pair_to_single = nn.Linear(pair_dim, single_dim)
        self.geometry_to_pair = nn.Linear(16, pair_dim)
        self.update = nn.Sequential(nn.LayerNorm(single_dim), nn.Linear(single_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, single_dim))
        self.backbone_delta = nn.Linear(single_dim, 3)
        self.chi_delta = nn.Linear(single_dim, 4)

    def forward(self, single, pair, geometry, mask):
        pair_value = pair + self.geometry_to_pair(geometry)
        aggregate = self.pair_to_single(pair_value).masked_fill(~mask[..., None], 0).sum(2)
        aggregate = aggregate / mask.sum(2, keepdim=True).clamp_min(1)
        node_mask = mask.any(dim=2)
        single = (single + self.update(single + aggregate)) * node_mask[..., None]
        return single, self.backbone_delta(single), self.chi_delta(single)


class MiniTorsionDiffusion(nn.Module):
    architecture_id = "lassodiff_mini_torsion_v2"
    schema_version = 2

    def __init__(self, single_dim=256, pair_dim=128, blocks=8, hidden_dim=256, heads=8, dropout=.05):
        super().__init__()
        self.single_dim, self.pair_dim = single_dim, pair_dim
        self.blocks = nn.ModuleList(_Block(single_dim, pair_dim, hidden_dim) for _ in range(blocks))
        self.time_projection = nn.Linear(single_dim, single_dim)
        self.state_projection = nn.Linear(14, single_dim)
        head_dim = 2 * single_dim + 14
        self.backbone_head = nn.Sequential(
            nn.LayerNorm(head_dim), nn.Linear(head_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        self.chi_head = nn.Sequential(
            nn.LayerNorm(head_dim), nn.Linear(head_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.geometry_step_scale = 0.05

    def forward(self, *, state_t: TorsionState, time: torch.Tensor, conditioning: MiniConditioning,
                token_mask: torch.Tensor, candidates: list):
        if state_t.backbone.ndim != 4:
            raise ValueError("state_t.backbone must be [B,Ns,L,3]")
        B, Ns, L, _ = state_t.backbone.shape
        if conditioning.single.shape[:2] != (B, L) or time.shape not in {(B,), (B, Ns)}:
            raise ValueError("conditioning/state/time shapes disagree")
        flat = B * Ns
        flat_state = TorsionState(
            state_t.backbone.reshape(flat, 1, L, 3), state_t.backbone_mask.reshape(flat, 1, L, 3),
            state_t.acceptor_chi.reshape(flat, 1, L, 4), state_t.acceptor_chi_mask.reshape(flat, 1, L, 4),
        )
        single = conditioning.single[:, None].expand(B, Ns, L, -1).reshape(flat, L, -1)
        pair = conditioning.pair[:, None].expand(B, Ns, L, L, -1).reshape(flat, L, L, -1)
        pair_mask = conditioning.pair_mask[:, None].expand(B, Ns, L, L).reshape(flat, L, L)
        flat_mask = token_mask[:, None].expand(B, Ns, L).reshape(flat, L)
        flat_candidates = [candidate for candidate in candidates for _ in range(Ns)]
        flat_sequences = [candidate.sequence for candidate in flat_candidates]
        state_features = torch.cat((torch.sin(flat_state.backbone[:, 0]), torch.cos(flat_state.backbone[:, 0]),
                                    torch.sin(flat_state.acceptor_chi[:, 0]), torch.cos(flat_state.acceptor_chi[:, 0])), dim=-1)
        single = single + self.state_projection(state_features)
        tf = time if time.ndim == 2 else time[:, None].expand(B, Ns)
        time_embedding = self.time_projection(
            _time_features(tf.reshape(-1), self.single_dim)
        ).reshape(flat, 1, -1)
        single = single + time_embedding
        running_backbone = flat_state.backbone[:, 0]
        running_chi = flat_state.acceptor_chi[:, 0]
        calls = 0
        for block in self.blocks:
            running_state = TorsionState(running_backbone[:, None], flat_state.backbone_mask,
                                         running_chi[:, None], flat_state.acceptor_chi_mask)
            coordinates = decode_lasso_core(running_state, sequences=flat_sequences,
                                            candidates=flat_candidates, token_mask=flat_mask)[:, 0]
            geometry = compute_dynamic_pair_geometry(coordinates, flat_mask)
            single, delta_backbone, delta_chi = block(single, pair, geometry, pair_mask)
            running_backbone = wrap_angle(running_backbone + self.geometry_step_scale * delta_backbone) * flat_state.backbone_mask[:, 0]
            running_chi = wrap_angle(running_chi + self.geometry_step_scale * delta_chi) * flat_state.acceptor_chi_mask[:, 0]
            calls += 1
        head_features = torch.cat(
            (single, state_features, time_embedding.expand(flat, L, -1)), dim=-1,
        )
        velocity = TorsionVelocity(
            self.backbone_head(head_features).reshape(B, Ns, L, 3) * state_t.backbone_mask,
            self.chi_head(head_features).reshape(B, Ns, L, 4) * state_t.acceptor_chi_mask,
        )
        return MiniTorsionOutput(velocity, calls)
