"""Hard rejection and transparent candidate ranking for V3 design mode."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .topology_checker import TopologyCheckResult


@dataclass(frozen=True)
class CandidateScore:
    hard_valid: torch.Tensor
    score: torch.Tensor
    rejection_reason: tuple[tuple[str, ...], ...]


def _normalise(value: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    return ((value.float() - minimum) / (maximum - minimum)).clamp(0.0, 1.0)


def score_candidates(
    check: TopologyCheckResult, *, structure_quality: torch.Tensor | None = None,
    instability: torch.Tensor | None = None, severe_clash_rate: float = .02,
    tail_clearance_min: float = 1.2,
) -> CandidateScore:
    """Return hard-valid mask, component score and explicit rejection reasons."""
    shape = check.topology_pass.shape
    device = check.topology_pass.device
    quality = torch.zeros(shape, device=device) if structure_quality is None else structure_quality.float()
    instability_value = torch.zeros(shape, device=device) if instability is None else instability.float()
    if quality.shape != shape or instability_value.shape != shape:
        raise ValueError("candidate score inputs must have shape [B,M]")
    finite = (
        torch.isfinite(check.iso_distance) & torch.isfinite(check.iso_angle_degrees)
        & torch.isfinite(check.iso_plane_distance) & torch.isfinite(check.clash_rate)
    )
    hard = (
        finite & check.ring_closed & check.threading_success & check.plug_retained
        & check.small_perturbation_stable & (check.tail_clearance >= tail_clearance_min)
        & (check.clash_rate <= severe_clash_rate)
    )
    geometry = (
        _normalise(check.iso_distance, 1.1, 1.7)
        * _normalise(check.iso_angle_degrees, 90.0, 150.0)
        * (1.0 - _normalise(check.iso_plane_distance, 0.0, .5))
    )
    score = (
        .45 * check.threading_confidence.float().clamp(0.0, 1.0)
        + .25 * geometry
        + .20 * quality.clamp(0.0, 1.0)
        - .07 * (check.clash_rate.float() / severe_clash_rate).clamp_min(0.0)
        - .03 * instability_value.clamp_min(0.0)
    )
    score = torch.where(hard, score, torch.full_like(score, float("-inf")))
    reasons = []
    for b in range(shape[0]):
        row = []
        for m in range(shape[1]):
            failed = []
            if not bool(finite[b, m]): failed.append("nonfinite")
            if not bool(check.ring_closed[b, m]): failed.append("closure")
            if not bool(check.threading_success[b, m]): failed.append("threading")
            if not bool(check.plug_retained[b, m]): failed.append("plug_retention")
            if not bool(check.small_perturbation_stable[b, m]): failed.append("unstable")
            if float(check.tail_clearance[b, m]) < tail_clearance_min: failed.append("tail_clearance")
            if float(check.clash_rate[b, m]) > severe_clash_rate: failed.append("clash")
            row.append(tuple(failed))
        reasons.append(tuple(row))
    return CandidateScore(hard, score, tuple(reasons))
