"""Candidate-preserving sampler for cached-OpenDDE LassoDiff V3."""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class V3SamplerOutput:
    coordinates: torch.Tensor
    model_calls: int


def sample_rectified_flow_v3(
    model, reasoning_state, candidates, atom_mask, *, steps: int = 40,
    initial_coordinates: torch.Tensor | None = None,
):
    if steps < 2:
        raise ValueError("V3 sampler requires steps >= 2")
    B, L = reasoning_state.token_mask.shape
    M, A = candidates.k.shape[1], atom_mask.shape[-1]
    expected = (B, M, L, A, 3)
    if initial_coordinates is None:
        x = torch.randn(expected, device=reasoning_state.single.device, dtype=reasoning_state.single.dtype)
    else:
        if tuple(initial_coordinates.shape) != expected:
            raise ValueError(f"initial coordinates must have shape {expected}")
        x = initial_coordinates.clone()
    if tuple(atom_mask.shape) != expected[:-1]:
        raise ValueError("V3 sampler requires candidate-specific atom_mask [B,M,L,A]")
    valid = atom_mask & reasoning_state.token_mask[:, None, :, None]
    dt = 1.0 / (steps - 1)
    for index in range(steps - 1):
        t = torch.full((B,), index * dt, device=x.device, dtype=x.dtype)
        output = model(reasoning_state, candidates, x, t, atom_mask)
        if output.velocity.shape != x.shape:
            raise RuntimeError("V3 sampler received collapsed candidate velocity")
        x = (x + dt * output.velocity) * valid[..., None].to(x.dtype)
        if not torch.isfinite(x).all():
            raise RuntimeError(f"V3 sampler produced non-finite coordinates at update {index}")
    return V3SamplerOutput(x, steps - 1)
