"""Projection-free lasso topology and reactive-geometry checker."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .evaluation_v2 import _linking_number
from .losses_lasso import ATOM_CA, ATOM_CISO, ATOM_N
from .topology_threading import hard_threading_check_ca


@dataclass
class TopologyCheckResult:
    iso_distance: torch.Tensor
    iso_angle_degrees: torch.Tensor
    iso_plane_distance: torch.Tensor
    iso_distance_valid: torch.Tensor
    iso_angle_valid: torch.Tensor
    iso_plane_valid: torch.Tensor
    ring_disk_crossing: torch.Tensor
    gauss_link_value: torch.Tensor
    threading_success: torch.Tensor
    clash_rate: torch.Tensor
    topology_pass: torch.Tensor
    checker_valid: torch.Tensor
    crossing_count: torch.Tensor
    signed_crossing: torch.Tensor
    threading_class: torch.Tensor
    threading_confidence: torch.Tensor
    ring_closed: torch.Tensor
    plug_retained: torch.Tensor
    tail_clearance: torch.Tensor
    small_perturbation_stable: torch.Tensor


def _angle(left, center, right):
    a, b = left - center, right - center
    cosine = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-8)
    return torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))


def _clash_rate(coords, atom_mask, token_mask):
    B, M, L, A, _ = coords.shape
    points = coords.reshape(B, M, L * A, 3)
    valid = (atom_mask & token_mask[:, None, :, None]).reshape(B, M, L * A)
    residue = torch.arange(L, device=coords.device).repeat_interleave(A)
    pair_allowed = (residue[:, None] - residue[None, :]).abs() > 1
    pair_allowed &= torch.triu(torch.ones(L * A, L * A, dtype=torch.bool, device=coords.device), diagonal=1)
    pair_valid = valid[..., :, None] & valid[..., None, :] & pair_allowed
    distance = torch.cdist(points.float(), points.float())
    clashes = (distance < 1.2) & pair_valid
    return clashes.sum((-2, -1)).float() / pair_valid.sum((-2, -1)).clamp(min=1)


def _plug_and_tail_geometry(ca, token_mask, candidates):
    B, M, L, _ = ca.shape
    plug_retained = torch.zeros((B, M), dtype=torch.bool, device=ca.device)
    tail_clearance = torch.full((B, M), float("inf"), dtype=ca.dtype, device=ca.device)
    for b in range(B):
        length = int(token_mask[b].sum())
        for m in range(M):
            if not bool(candidates.candidate_mask[b, m]):
                continue
            k, p = int(candidates.k[b, m]), int(candidates.p[b, m])
            if p >= length:
                continue
            ring = ca[b, m, :k + 1]
            plug_distance = torch.cdist(ca[b, m, p:p + 1].float(), ring.float()).min()
            # This is an intentionally permissive diagnostic threshold.  The
            # locked truth set, not this proxy, calibrates release scoring.
            plug_retained[b, m] = plug_distance <= 6.0
            if p + 1 < length:
                tail_clearance[b, m] = torch.cdist(ca[b, m, p + 1:length].float(), ring.float()).min()
    return plug_retained, tail_clearance


@torch.no_grad()
def strict_topology_check(coords, atom_mask, candidates, *, token_mask):
    if coords.ndim != 5 or atom_mask.shape != coords.shape[:-1]:
        raise ValueError("topology checker requires [B,M,L,A,3] coordinates and candidate atom mask")
    B, M, L, A, _ = coords.shape
    candidates.validate(L)
    batch = torch.arange(B, device=coords.device)[:, None].expand(B, M)
    candidate = torch.arange(M, device=coords.device)[None].expand(B, M)
    acceptor = candidates.acceptor_index
    nterm = coords[:, :, 0, ATOM_N]
    ciso = coords[batch, candidate, acceptor, ATOM_CISO]
    oxygen1 = coords[batch, candidate, acceptor, 5]
    acceptor_ca = coords[batch, candidate, acceptor, ATOM_CA]
    iso_distance = (nterm - ciso).norm(dim=-1)
    iso_angle = _angle(nterm, ciso, oxygen1)
    plane_normal = torch.linalg.cross(oxygen1 - ciso, acceptor_ca - ciso, dim=-1)
    iso_plane = ((nterm - ciso) * plane_normal).sum(-1).abs() / plane_normal.norm(dim=-1).clamp_min(1e-8)
    # A formed isopeptide amide has one carbonyl oxygen; requiring the second
    # carboxylate oxygen made every real target invalid by construction.
    side_valid = (
        atom_mask[batch, candidate, acceptor, ATOM_CA]
        & atom_mask[batch, candidate, acceptor, ATOM_CISO]
        & atom_mask[batch, candidate, acceptor, 5]
        & atom_mask[:, :, 0, ATOM_N]
    )
    iso_distance_valid = side_valid & (iso_distance >= 1.1) & (iso_distance <= 1.7)
    iso_angle_valid = side_valid & (iso_angle >= 90) & (iso_angle <= 150)
    iso_plane_valid = side_valid & (iso_plane <= .5)
    ca = coords[..., ATOM_CA, :]
    threading_check = hard_threading_check_ca(
        ca, token_mask, candidates, atom_mask[..., ATOM_CA],
    )
    # A valid lasso candidate has exactly one plug-to-tail crossing.  A
    # double crossing is not accepted merely because a continuous proxy is
    # non-zero.  Gauss link remains diagnostic only.
    crossing = threading_check.crossing_count > 0
    link = _linking_number(coords, token_mask, candidates)
    threading = threading_check.valid & (threading_check.crossing_count == 1)
    clash = _clash_rate(coords, atom_mask, token_mask)
    valid = candidates.candidate_mask.bool() & threading_check.valid
    ring_closed = side_valid & iso_distance_valid & iso_angle_valid & iso_plane_valid
    plug_retained, tail_clearance = _plug_and_tail_geometry(ca, token_mask, candidates)
    # Evaluate a deterministic tiny perturbation to expose boundary-sensitive
    # checker decisions.  This is an evaluation-only stability signal.
    perturbation = torch.zeros_like(ca)
    perturbation[..., 0] = 1e-4
    perturbation[..., 1] = -1e-4
    perturbed = hard_threading_check_ca(
        ca + perturbation, token_mask, candidates, atom_mask[..., ATOM_CA],
    )
    stable = threading_check.valid & perturbed.valid & threading_check.crossing_count.eq(perturbed.crossing_count)
    topology_pass = valid & iso_distance_valid & iso_angle_valid & iso_plane_valid & threading & (clash <= .02)
    return TopologyCheckResult(
        iso_distance, iso_angle, iso_plane, iso_distance_valid & valid, iso_angle_valid & valid,
        iso_plane_valid & valid, crossing & valid, link, threading & valid, clash, topology_pass,
        threading_check.valid & candidates.candidate_mask.bool(), threading_check.crossing_count,
        threading_check.signed_crossing, threading_check.threading_class,
        threading_check.confidence, ring_closed & valid, plug_retained & valid,
        tail_clearance, stable & valid,
    )
