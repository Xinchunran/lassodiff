"""Circular rectified flow utilities for torsion angles."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .torsion_state import TorsionState, TorsionVelocity, validate_torsion_state


def wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def shortest_angular_difference(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(target - source), torch.cos(target - source))


@dataclass
class TorsionFlow:
    state_t: TorsionState
    velocity: TorsionVelocity


def _expand_time(time: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if time.ndim != 1 or time.shape[0] != value.shape[0]:
        raise ValueError("flow time must have shape [B]")
    return time.reshape(time.shape[0], *([1] * (value.ndim - 1)))


def interpolate_torsion_state(source: TorsionState, target: TorsionState, time: torch.Tensor) -> TorsionFlow:
    validate_torsion_state(source)
    validate_torsion_state(target)
    if source.backbone.shape != target.backbone.shape or source.acceptor_chi.shape != target.acceptor_chi.shape:
        raise ValueError("source and target torsion shapes differ")
    tb = _expand_time(time, source.backbone)
    tc = _expand_time(time, source.acceptor_chi)
    vb = shortest_angular_difference(target.backbone, source.backbone)
    vc = shortest_angular_difference(target.acceptor_chi, source.acceptor_chi)
    state = TorsionState(
        wrap_angle(source.backbone + tb * vb) * source.backbone_mask,
        source.backbone_mask,
        wrap_angle(source.acceptor_chi + tc * vc) * source.acceptor_chi_mask,
        source.acceptor_chi_mask,
    )
    return TorsionFlow(state, TorsionVelocity(vb * source.backbone_mask, vc * source.acceptor_chi_mask))


def estimate_torsion_endpoint(state_t: TorsionState, velocity: TorsionVelocity, time: torch.Tensor) -> TorsionState:
    tb = _expand_time(time, state_t.backbone)
    tc = _expand_time(time, state_t.acceptor_chi)
    return TorsionState(
        wrap_angle(state_t.backbone + (1.0 - tb) * velocity.backbone) * state_t.backbone_mask,
        state_t.backbone_mask,
        wrap_angle(state_t.acceptor_chi + (1.0 - tc) * velocity.acceptor_chi) * state_t.acceptor_chi_mask,
        state_t.acceptor_chi_mask,
    )
