"""Masked, rigid-frame-aware rectified flow utilities for Mini."""
from __future__ import annotations

import torch


def _center(x, mask):
    weight = mask[..., None].to(x.dtype)
    center = (x * weight).sum((-3, -2), keepdim=True) / weight.sum((-3, -2), keepdim=True).clamp_min(1)
    return (x - center) * weight


def _align_target(source, target, mask):
    shape = source.shape
    source_f = source.reshape(-1, shape[-3] * shape[-2], 3)
    target_f = target.reshape_as(source_f)
    weight = mask.reshape(-1, shape[-3] * shape[-2]).to(source.dtype)
    covariance = (target_f * weight[..., None]).transpose(-2, -1) @ source_f
    u, _s, vh = torch.linalg.svd(covariance)
    rotation = vh.transpose(-2, -1) @ u.transpose(-2, -1)
    determinant = torch.linalg.det(rotation)
    correction = torch.ones((*determinant.shape, 3), dtype=source.dtype, device=source.device)
    correction[..., -1] = determinant
    rotation = vh.transpose(-2, -1) @ torch.diag_embed(correction) @ u.transpose(-2, -1)
    return (target_f @ rotation.transpose(-2, -1)).reshape(shape) * mask[..., None]


def align_target_to_source(source, target, mask):
    """Center and Kabsch-align target to source over the supplied atom mask."""
    if source.shape != target.shape or mask.shape != source.shape[:-1]:
        raise ValueError("alignment tensors/mask have incompatible shapes")
    source_centered = _center(source, mask)
    target_centered = _center(target, mask)
    return _align_target(source_centered, target_centered, mask)


def flow_interpolate(x0, x1, t, atom_mask, *, align_target: bool = True):
    if x0.shape != x1.shape or atom_mask.shape != x0.shape[:-1]:
        raise ValueError("flow tensors/mask have incompatible shapes")
    if t.shape != (x0.shape[0],):
        raise ValueError("flow time must have shape [B]")
    x0 = _center(x0, atom_mask)
    x1 = _center(x1, atom_mask)
    if align_target:
        x1 = _align_target(x0, x1, atom_mask)
    expand = (slice(None),) + (None,) * (x0.ndim - 1)
    x_t = (1 - t[expand]) * x0 + t[expand] * x1
    return x_t * atom_mask[..., None], (x1 - x0) * atom_mask[..., None], x1
