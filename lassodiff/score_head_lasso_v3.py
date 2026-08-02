from __future__ import annotations

import math

import torch
import torch.nn as nn

from .geometry_features import DynamicGeometryEncoder, GeometryProjection


class FourierFlowEmbedding(nn.Module):
    def __init__(self, output_dim: int, frequencies: int = 16):
        super().__init__()
        self.register_buffer("frequency", torch.exp(torch.linspace(0.0, math.log(1000.0), frequencies)), persistent=False)
        self.net = nn.Sequential(nn.Linear(2 * frequencies, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim))

    def forward(self, t):
        ratio = (1.0 - t).clamp(min=1e-4) / t.clamp(min=1e-4)
        log_noise = ratio.log() / 4.0
        phase = log_noise[..., None] * self.frequency.to(log_noise.dtype)
        return self.net(torch.cat([phase.sin(), phase.cos()], dim=-1))


class _InvariantPairBlock(nn.Module):
    def __init__(self, c_a: int, c_z: int, n_heads: int):
        super().__init__()
        if c_a % n_heads:
            raise ValueError("c_a must be divisible by n_heads")
        self.n_heads, self.head_dim = n_heads, c_a // n_heads
        self.norm = nn.LayerNorm(c_a)
        self.z_norm = nn.LayerNorm(c_z)
        self.q, self.k, self.v = nn.Linear(c_a, c_a), nn.Linear(c_a, c_a), nn.Linear(c_a, c_a)
        self.bias = nn.Linear(c_z, n_heads)
        self.out = nn.Linear(c_a, c_a)
        self.transition = nn.Sequential(nn.LayerNorm(c_a), nn.Linear(c_a, 4 * c_a), nn.GELU(), nn.Linear(4 * c_a, c_a))

    def forward(self, a, z, pair_mask):
        B, M, L, C = a.shape
        norm = self.norm(a)
        q = self.q(norm).reshape(B, M, L, self.n_heads, self.head_dim)
        k = self.k(norm).reshape(B, M, L, self.n_heads, self.head_dim)
        v = self.v(norm).reshape(B, M, L, self.n_heads, self.head_dim)
        logits = torch.einsum("bmihd,bmjhd->bmhij", q, k) / math.sqrt(self.head_dim)
        logits = logits + self.bias(self.z_norm(z)).permute(0, 1, 4, 2, 3)
        logits = logits.masked_fill(~pair_mask[:, :, None], torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(pair_mask[:, :, None], weights, torch.zeros_like(weights))
        update = torch.einsum("bmhij,bmjhd->bmihd", weights, v).reshape(B, M, L, C)
        a = a + self.out(update)
        return a + self.transition(a)


class EquivariantLassoDiffusionV3(nn.Module):
    """Invariant scalar reasoning with an SE(3)-equivariant vector decoder."""

    def __init__(self, c_s: int, c_z: int, c_a: int, n_blocks: int, n_heads: int, n_atom: int = 7):
        super().__init__()
        self.n_atom = n_atom
        self.geometry_encoder = DynamicGeometryEncoder()
        self.geometry_projection = GeometryProjection(self.geometry_encoder.output_dim, c_z)
        self.single_projection = nn.Linear(c_s, c_a)
        self.internal_distance_projection = nn.Linear(n_atom, c_a)
        self.noise_embedding = FourierFlowEmbedding(c_a)
        self.blocks = nn.ModuleList([_InvariantPairBlock(c_a, c_z, n_heads) for _ in range(n_blocks)])
        self.local_scale = nn.Linear(c_a, n_atom)
        self.pair_scale = nn.Linear(c_z, n_atom)

    def forward(self, x_t, t, s, z_static, token_mask, atom_mask, pair_mask, trace=None, candidates=None):
        B, M, L, A, _ = x_t.shape
        if A != self.n_atom or atom_mask.shape != (B, M, L, A):
            raise ValueError("V3 x_t/atom_mask shape mismatch")
        valid = atom_mask & token_mask[:, None, :, None]
        x_geometry = x_t.float()
        weight = valid.to(x_geometry.dtype)
        centroid = (x_geometry * weight[..., None]).sum(dim=-2) / weight.sum(dim=-1, keepdim=True).clamp(min=1.0)
        centered = x_geometry - centroid[..., None, :]
        internal_distance = torch.linalg.vector_norm(centered, dim=-1) * weight
        if t.ndim == 1:
            t = t[:, None].expand(B, M)
        a = self.single_projection(s) + self.internal_distance_projection(internal_distance)
        a = a + self.noise_embedding(t)[:, :, None]
        z = z_static
        for block in self.blocks:
            geometry = self.geometry_encoder(x_t, atom_mask, token_mask, candidates=candidates)
            z = z_static + self.geometry_projection(geometry, pair_mask)
            a = block(a, z, pair_mask)
            if trace is not None:
                trace.geometry_calls += 1
                trace.diffusion_calls += 1

        local = self.local_scale(a).float()[..., None] * centered
        displacement = centroid[:, :, None, :, :] - centroid[:, :, :, None, :]
        distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True).clamp(min=1e-6)
        direction = displacement / distance
        pair_coeff = self.pair_scale(z).float() * pair_mask[..., None].to(torch.float32)
        pair_vector = torch.einsum("bmija,bmijc->bmiac", pair_coeff, direction)
        denominator = pair_mask.sum(dim=-1).clamp(min=1)[..., None, None]
        velocity = local + pair_vector / denominator
        return velocity * valid[..., None].to(velocity.dtype)
