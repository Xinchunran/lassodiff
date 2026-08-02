"""Differentiable candidate-specific endpoint topology objective for V3."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .losses_lasso import (
    ATOM_C, ATOM_CA, ATOM_CISO, ATOM_N, ATOM_O, ATOM_O1,
)
from .topology_threading import hard_threading_check_ca, soft_threading_score_ca


@dataclass(frozen=True)
class TopologyLossWeights:
    flow: float = 1.0
    bond: float = 1.0
    iso_distance: float = 10.0
    iso_angle: float = 2.0
    iso_plane: float = 2.0
    threading: float = 2.0


@dataclass
class CandidateTopologyLoss:
    total: torch.Tensor
    flow: torch.Tensor
    bond: torch.Tensor
    iso_distance: torch.Tensor
    iso_angle: torch.Tensor
    iso_plane: torch.Tensor
    threading: torch.Tensor
    endpoint: torch.Tensor


def endpoint_from_velocity(x_t: torch.Tensor, velocity: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    if x_t.shape != velocity.shape or x_t.ndim != 5:
        raise ValueError("V3 endpoint reconstruction requires [B,M,L,A,3] tensors")
    if time.ndim != 1 or time.shape[0] != x_t.shape[0]:
        raise ValueError("V3 endpoint reconstruction requires time [B]")
    scale = (1.0 - time.float())[:, None, None, None, None]
    return x_t.float() + scale * velocity.float()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    dimensions = tuple(range(2, value.ndim))
    return (value * weight).sum(dim=dimensions) / weight.sum(dim=dimensions).clamp_min(1.0)


def _bond_loss(pred: torch.Tensor, target: torch.Tensor, token_mask, atom_mask):
    terms, masks = [], []

    def add(res_a, atom_a, res_b, atom_b, valid):
        pred_distance = (pred[..., res_a, atom_a, :] - pred[..., res_b, atom_b, :]).norm(dim=-1)
        target_distance = (target[..., res_a, atom_a, :] - target[..., res_b, atom_b, :]).norm(dim=-1)
        terms.append(F.smooth_l1_loss(pred_distance, target_distance, reduction="none", beta=0.1))
        masks.append(valid & atom_mask[..., res_a, atom_a] & atom_mask[..., res_b, atom_b])

    B, M, L = pred.shape[:3]
    token = token_mask[:, None].expand(B, M, L)
    index = torch.arange(L, device=pred.device)
    add(index, ATOM_N, index, ATOM_CA, token)
    add(index, ATOM_CA, index, ATOM_C, token)
    add(index, ATOM_C, index, ATOM_O, token)
    if L > 1:
        add(index[:-1], ATOM_C, index[1:], ATOM_N, token[..., :-1] & token[..., 1:])
    numerator = sum((term * mask.to(term.dtype)).sum(-1) for term, mask in zip(terms, masks))
    denominator = sum(mask.sum(-1) for mask in masks).clamp_min(1)
    return numerator / denominator


def _reactive_geometry(coords, candidates):
    B, M, L = coords.shape[:3]
    batch = torch.arange(B, device=coords.device)[:, None].expand(B, M)
    candidate = torch.arange(M, device=coords.device)[None].expand(B, M)
    acceptor = candidates.acceptor_index.clamp(0, L - 1)
    nterm = coords[:, :, 0, ATOM_N]
    ciso = coords[batch, candidate, acceptor, ATOM_CISO]
    oxygen = coords[batch, candidate, acceptor, ATOM_O1]
    acceptor_ca = coords[batch, candidate, acceptor, ATOM_CA]
    first, second = nterm - ciso, oxygen - ciso
    distance = first.norm(dim=-1)
    cosine = (first * second).sum(-1) / (first.norm(dim=-1) * second.norm(dim=-1)).clamp_min(1e-8)
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    normal = torch.linalg.cross(second, acceptor_ca - ciso, dim=-1)
    plane = (first * normal).sum(-1).abs() / normal.norm(dim=-1).clamp_min(1e-8)
    return distance, angle, plane


def topology_supervised_candidate_loss(
    velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    x_t: torch.Tensor,
    time: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    candidates,
    weights: TopologyLossWeights,
) -> CandidateTopologyLoss:
    if target.shape != velocity.shape or atom_mask.shape != target.shape[:-1]:
        raise ValueError("topology objective requires candidate-specific target/mask tensors")
    B, M, L = target.shape[:3]
    candidates.validate(L)
    endpoint = endpoint_from_velocity(x_t, velocity, time)
    target = target.float()
    valid_atom = atom_mask & token_mask[:, None, :, None] & candidates.candidate_mask[:, :, None, None]
    # Match the historical candidate MSE scale: mean per coordinate, not the
    # sum over xyz, so topology weights remain interpretable.
    flow_error = (velocity.float() - target_velocity.float()).square().sum(-1) / 3.0
    flow = _masked_mean(flow_error, valid_atom)
    bond = _bond_loss(endpoint, target, token_mask, atom_mask)
    pred_distance, pred_angle, pred_plane = _reactive_geometry(endpoint, candidates)
    target_distance, target_angle, target_plane = _reactive_geometry(target, candidates)
    valid = candidates.candidate_mask.to(endpoint.dtype)
    distance_target = F.smooth_l1_loss(pred_distance, target_distance, reduction="none", beta=0.1)
    distance_band = F.relu(1.1 - pred_distance).square() + F.relu(pred_distance - 1.7).square()
    iso_distance = (distance_target + distance_band) * valid
    angle_target = F.smooth_l1_loss(pred_angle / 30.0, target_angle / 30.0, reduction="none", beta=0.1)
    angle_band = F.relu((90.0 - pred_angle) / 30.0).square() + F.relu((pred_angle - 150.0) / 30.0).square()
    iso_angle = (angle_target + angle_band) * valid
    plane_target = F.smooth_l1_loss(pred_plane, target_plane, reduction="none", beta=0.1)
    iso_plane = (plane_target + F.relu(pred_plane - 0.5).square()) * valid
    # Target validity and the continuous target score come from the same
    # p..tail segment/surface geometry as strict evaluation.  Matching the
    # target's soft score keeps the exact endpoint at zero loss while the hard
    # checker independently verifies that the target has one signed crossing.
    target_threading = hard_threading_check_ca(
        target[..., ATOM_CA, :], token_mask, candidates, atom_mask[..., ATOM_CA],
    )
    predicted_score = soft_threading_score_ca(
        endpoint[..., ATOM_CA, :], token_mask, candidates, atom_mask[..., ATOM_CA],
    )
    target_score = soft_threading_score_ca(
        target[..., ATOM_CA, :], token_mask, candidates, atom_mask[..., ATOM_CA],
    ).detach()
    threading_valid = valid * target_threading.valid.to(valid.dtype)
    threading = F.smooth_l1_loss(
        predicted_score, target_score,
        reduction="none", beta=0.1,
    ) * threading_valid
    total = (
        float(weights.flow) * flow
        + float(weights.bond) * bond
        + float(weights.iso_distance) * iso_distance
        + float(weights.iso_angle) * iso_angle
        + float(weights.iso_plane) * iso_plane
        + float(weights.threading) * threading
    )
    total = torch.where(candidates.candidate_mask, total, torch.zeros_like(total))
    return CandidateTopologyLoss(total, flow, bond, iso_distance, iso_angle, iso_plane, threading, endpoint)
