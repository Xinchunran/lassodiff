"""Invariant dynamic pair geometry computed from the current backbone."""
from __future__ import annotations

import torch


def _frames(coordinates: torch.Tensor) -> torch.Tensor:
    n, ca, c = coordinates[..., 0, :], coordinates[..., 1, :], coordinates[..., 2, :]
    x = (c - ca); x = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    y = (n - ca); y = y - (y * x).sum(-1, keepdim=True) * x
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    z = torch.linalg.cross(x, y, dim=-1)
    return torch.stack((x, y, z), -1)


def compute_dynamic_pair_geometry(coordinates: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    if coordinates.ndim != 4 or coordinates.shape[-2] < 3 or coordinates.shape[-1] != 3:
        raise ValueError("coordinates must be [B,L,A,3]")
    ca = coordinates[..., 1, :]
    frames = _frames(coordinates)
    delta = ca[:, None, :, :] - ca[:, :, None, :]
    # Apply each residue's inverse local frame to the i->j vector.
    local_direction = torch.einsum("blag,bljg->blja", frames.transpose(-1, -2), delta)
    orientation = frames.transpose(-1, -2)[:, :, None] @ frames[:, None, :]
    distance = delta.norm(dim=-1, keepdim=True)
    separation = torch.arange(ca.shape[1], device=ca.device)[None, :, None] - torch.arange(ca.shape[1], device=ca.device)[None, None, :]
    valid = token_mask[:, :, None] & token_mask[:, None, :]
    features = torch.cat((distance, local_direction, orientation.reshape(*orientation.shape[:3], 9),
                          separation[..., None].to(ca.dtype) / max(ca.shape[1], 1),
                          separation.abs()[..., None].to(ca.dtype) / max(ca.shape[1], 1),
                          valid[..., None].to(ca.dtype)), -1)
    return features * valid[..., None].to(features.dtype)
