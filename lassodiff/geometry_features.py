"""Dynamic coordinate-derived, translation-invariant pair geometry."""
from __future__ import annotations

import torch
import torch.nn as nn


class DynamicGeometryEncoder(nn.Module):
    def __init__(self, n_rbf: int = 16, cutoff: float = 24.0):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf)
        self.register_buffer("centers", centers, persistent=False)
        self.gamma = float((n_rbf - 1) / cutoff) ** 2
        # RBF/min-distance/valid-fraction + local direction (3) + relative
        # frame diagonal (3) + reactive/topology pair types (3).
        self.output_dim = n_rbf + 11

    def forward(self, x_t, atom_mask, token_mask, candidates=None):
        """Return scalar features [B,M,L,L,F]; ignores invalid/padded atoms."""
        if x_t.ndim != 5:
            raise ValueError(f"expected x_t [B,M,L,A,3], got {tuple(x_t.shape)}")
        B, M, L, A, _ = x_t.shape
        # Vector algebra remains fp32 under FSDP/autocast.  Norm/cross/SVD-like
        # operations may selectively promote bf16 inputs; making the contract
        # explicit avoids mixed-dtype vector operations and improves stability.
        x_t = x_t.float()
        if atom_mask.shape == (B, L, A):
            atom_mask = atom_mask[:, None].expand(B, M, L, A)
        elif atom_mask.shape != (B, M, L, A):
            raise ValueError("atom_mask must have shape [B,L,A] or [B,M,L,A]")
        valid = atom_mask & token_mask[:, None, :, None]
        weight = valid.to(x_t.dtype)
        centroid = (x_t * weight[..., None]).sum(dim=-2) / weight.sum(dim=-1, keepdim=True).clamp(min=1.0)
        # A centroid pair distance is rotation and translation invariant.  The
        # min atom distance adds local chemistry while preserving invariance.
        displacement = centroid[:, :, :, None] - centroid[:, :, None, :]
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        atom_delta = x_t[:, :, :, None, :, None, :] - x_t[:, :, None, :, None, :, :]
        atom_distance = torch.linalg.vector_norm(atom_delta, dim=-1)
        atom_pair_mask = valid[:, :, :, None, :, None] & valid[:, :, None, :, None, :]
        min_atom_distance = atom_distance.masked_fill(~atom_pair_mask, torch.finfo(atom_distance.dtype).max).amin(dim=(-2, -1))
        pair_mask = token_mask[:, None, :, None] & token_mask[:, None, None, :]
        min_atom_distance = torch.where(pair_mask, min_atom_distance, torch.zeros_like(min_atom_distance))
        rbf = torch.exp(-self.gamma * (distance[..., None] - self.centers.to(distance.dtype)) ** 2)
        valid_fraction = atom_pair_mask.to(x_t.dtype).mean(dim=(-2, -1)).expand_as(distance)
        # Local residue frame from N--CA--C.  Dot products with this frame are
        # rotation invariant while retaining directional geometry.
        origin = x_t[..., 1, :]
        axis_x = x_t[..., 2, :] - origin
        axis_x = axis_x / axis_x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        provisional = x_t[..., 0, :] - origin
        axis_z = torch.linalg.cross(axis_x, provisional, dim=-1)
        axis_z = axis_z / axis_z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        axis_y = torch.linalg.cross(axis_z, axis_x, dim=-1)
        frame = torch.stack([axis_x, axis_y, axis_z], dim=-2)
        unit = displacement / distance[..., None].clamp_min(1e-6)
        local_direction = torch.einsum("bmijc,bmidc->bmijd", unit, frame)
        frame_diagonal = torch.einsum("bmidc,bmjdc->bmijd", frame, frame)
        reactive = torch.zeros((*distance.shape, 3), device=x_t.device, dtype=x_t.dtype)
        if candidates is not None:
            if tuple(candidates.k.shape) != (B, M):
                raise ValueError("geometry candidates must match [B,M]")
            idx = torch.arange(L, device=x_t.device)[None, None]
            nterm = idx == 0
            acceptor = idx == candidates.acceptor_index[..., None]
            plug = idx == candidates.p[..., None]
            ring = idx <= candidates.k[..., None]
            tail = idx > candidates.p[..., None]
            reactive[..., 0] = (nterm[..., :, None] & acceptor[..., None, :] | acceptor[..., :, None] & nterm[..., None, :])
            reactive[..., 1] = (plug[..., :, None] & ring[..., None, :] | ring[..., :, None] & plug[..., None, :])
            reactive[..., 2] = (tail[..., :, None] & ring[..., None, :] | ring[..., :, None] & tail[..., None, :])
        output = torch.cat([
            rbf, min_atom_distance[..., None] / 24.0, valid_fraction[..., None],
            local_direction, frame_diagonal, reactive,
        ], dim=-1)
        return output * pair_mask[..., None].to(output.dtype)


class GeometryProjection(nn.Module):
    def __init__(self, feature_dim: int, c_z: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, c_z), nn.GELU(), nn.Linear(c_z, c_z))

    def forward(self, geometry, pair_mask):
        return self.net(geometry) * pair_mask[..., None].to(geometry.dtype)
