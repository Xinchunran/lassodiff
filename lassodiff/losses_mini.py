"""Minimal core losses: flow, peptide bonds, formed amide, clash and exclusivity."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .atom_schema_lasso import ATOM_C, ATOM_CA, ATOM_CB, ATOM_CISO, ATOM_N, ATOM_OISO
from .validation.threading_mini import soft_topology_surrogate_ca


@dataclass(frozen=True)
class MiniLossBreakdown:
    total: torch.Tensor
    flow: torch.Tensor
    backbone: torch.Tensor
    iso_amide: torch.Tensor
    core_clash: torch.Tensor
    crossing_count: torch.Tensor
    exactly_one: torch.Tensor


def _band(value, lower, upper):
    return F.relu(lower - value).square() + F.relu(value - upper).square()


def _angle(a, center, b):
    left, right = a - center, b - center
    cosine = (left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1)).clamp_min(1e-8)
    return torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))


def iso_amide_loss(core, atom_mask, candidates, *, glu_side_anchor=None):
    B, M, L, _A, _ = core.shape
    batch = torch.arange(B, device=core.device)[:, None]
    member = torch.arange(M, device=core.device)[None]
    k = candidates.acceptor_index
    nterm = core[:, :, 0, ATOM_N]
    ciso = core[batch, member, k, ATOM_CISO]
    oiso = core[batch, member, k, ATOM_OISO]
    is_glu = candidates.acceptor_type is not None and candidates.acceptor_type.bool()
    side = core[batch, member, k, ATOM_CB]
    if is_glu is not False:
        if glu_side_anchor is None and bool(is_glu.any()):
            # Core-7 does not contain Glu CG.  Construct the real S atom on
            # the CB->CISO path; strict all-heavy validation still requires
            # an explicit CG and never substitutes CB.
            direction = ciso - side
            constructed_cg = side + 1.52 * direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            side = torch.where(is_glu[..., None], constructed_cg, side)
        elif glu_side_anchor is not None:
            side = torch.where(is_glu[..., None], glu_side_anchor, side)
    valid = candidates.candidate_mask & atom_mask[:, :, 0, ATOM_N]
    valid &= atom_mask[batch, member, k, ATOM_CISO] & atom_mask[batch, member, k, ATOM_OISO]
    distance = _band((nterm - ciso).norm(dim=-1), 1.25, 1.65)
    oxygen_angle = _band(_angle(oiso, ciso, nterm), 105.0, 135.0)
    side_angle = _band(_angle(side, ciso, nterm), 105.0, 135.0)
    normal = torch.linalg.cross(oiso - ciso, side - ciso, dim=-1)
    plane = (((nterm - ciso) * normal).sum(-1) / normal.norm(dim=-1).clamp_min(1e-8)).square()
    value = distance + .01 * (oxygen_angle + side_angle) + plane
    return value[valid].mean() if bool(valid.any()) else core.sum() * 0


def mini_core_loss(
    predicted_velocity, target_velocity, x_t, t, target, token_mask, atom_mask, candidates,
    *, glu_side_anchor=None, weights=(1.0, .2, 2.0, .05, .2, .2),
):
    endpoint = x_t + (1.0 - t[:, None, None, None, None]) * predicted_velocity
    valid = atom_mask & token_mask[:, None, :, None] & candidates.candidate_mask[:, :, None, None]
    flow = ((predicted_velocity - target_velocity).square().sum(-1) * valid).sum() / (valid.sum().clamp_min(1) * 3)
    peptide = (endpoint[:, :, :-1, ATOM_C] - endpoint[:, :, 1:, ATOM_N]).norm(dim=-1)
    peptide_valid = valid[:, :, :-1, ATOM_C] & valid[:, :, 1:, ATOM_N]
    backbone = ((peptide - 1.329).square() * peptide_valid).sum() / peptide_valid.sum().clamp_min(1)
    iso = iso_amide_loss(endpoint, atom_mask, candidates, glu_side_anchor=glu_side_anchor)
    points = endpoint.reshape(*endpoint.shape[:2], -1, 3)
    point_mask = valid.reshape(*valid.shape[:2], -1)
    distance = torch.cdist(points.float(), points.float())
    eye = torch.eye(distance.shape[-1], dtype=torch.bool, device=distance.device)
    pairs = point_mask[..., :, None] & point_mask[..., None, :] & ~eye
    clash = (F.relu(1.2 - distance).square() * pairs).sum() / pairs.sum().clamp_min(1)
    topology = soft_topology_surrogate_ca(endpoint[..., ATOM_CA, :], token_mask, candidates, atom_mask[..., ATOM_CA])
    topology_valid = topology.valid & candidates.candidate_mask
    count = ((topology.expected_count - 1).square()[topology_valid].mean() if bool(topology_valid.any()) else endpoint.sum() * 0)
    exact = ((-torch.log(topology.probability_exactly_one.clamp_min(1e-6)))[topology_valid].mean() if bool(topology_valid.any()) else endpoint.sum() * 0)
    components = (flow, backbone, iso, clash, count, exact)
    total = sum(weight * component for weight, component in zip(weights, components))
    return MiniLossBreakdown(total, *components)
