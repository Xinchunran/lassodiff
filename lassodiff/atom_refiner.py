"""Small equivariant all-heavy-atom refiner with bounded total displacement."""
from __future__ import annotations

import torch
import torch.nn as nn


class _EGNNRefinerBlock(nn.Module):
    def __init__(self, hidden_dim: int, max_neighbors: int):
        super().__init__()
        self.max_neighbors = int(max_neighbors)
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.coordinate_weight = nn.Linear(hidden_dim, 1)
        self.node_update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, h, x, mask, covalent):
        size = x.shape[1]
        distance2_full = torch.cdist(x.float(), x.float()).square()
        pair_valid = mask[:, :, None] & mask[:, None, :]
        pair_valid &= ~torch.eye(size, dtype=torch.bool, device=x.device)[None]
        ranking = distance2_full.masked_fill(~pair_valid, torch.inf)
        neighbor_count = min(self.max_neighbors, max(size - 1, 1))
        neighbor_distance2, neighbor = torch.topk(ranking, neighbor_count, dim=-1, largest=False)
        batch = torch.arange(x.shape[0], device=x.device)[:, None, None]
        h_j = h[batch, neighbor]
        x_j = x[batch, neighbor]
        h_i = h[:, :, None].expand(-1, -1, neighbor_count, -1)
        relative = x[:, :, None] - x_j
        distance2 = neighbor_distance2.clamp_max(100.0)[..., None].to(x.dtype)
        pair_mask = torch.isfinite(neighbor_distance2) & mask[:, :, None]
        covalent_neighbor = torch.gather(covalent, 2, neighbor)
        message = self.message(torch.cat((
            h_i, h_j, distance2, covalent_neighbor[..., None].to(distance2.dtype),
        ), dim=-1)) * pair_mask[..., None]
        denominator = pair_mask.sum(-1, keepdim=True).clamp_min(1).to(x.dtype)
        delta = (relative * self.coordinate_weight(message)).sum(2) / denominator
        h = h + self.node_update(torch.cat((h, message.sum(2) / denominator), dim=-1))
        return h * mask[..., None], x + delta * mask[..., None]


class MiniAtomRefiner(nn.Module):
    def __init__(
        self, hidden_dim: int = 96, layers: int = 4,
        max_displacement: float = .75, max_neighbors: int = 32,
    ):
        super().__init__()
        self.residue_embedding = nn.Embedding(21, hidden_dim)
        self.atom_embedding = nn.Embedding(14, hidden_dim)
        self.blocks = nn.ModuleList(_EGNNRefinerBlock(hidden_dim, max_neighbors) for _ in range(layers))
        self.max_displacement = float(max_displacement)

    def forward(self, coordinates, aa_ids, atom_mask, covalent_adjacency=None):
        if coordinates.ndim != 4 or coordinates.shape[-2:] != (14, 3):
            raise ValueError("refiner coordinates must have shape [B,L,14,3]")
        B, L, A, _ = coordinates.shape
        if aa_ids.shape != (B, L) or atom_mask.shape != (B, L, A):
            raise ValueError("refiner aa_ids/atom_mask shape mismatch")
        flat_mask = atom_mask.reshape(B, L * A)
        atom_ids = torch.arange(A, device=coordinates.device)[None, None].expand(B, L, A)
        h = self.residue_embedding(aa_ids)[:, :, None] + self.atom_embedding(atom_ids)
        h = h.reshape(B, L * A, -1) * flat_mask[..., None]
        x0 = coordinates.reshape(B, L * A, 3)
        x = x0
        if covalent_adjacency is None:
            covalent_adjacency = torch.zeros((B, L * A, L * A), dtype=torch.bool, device=x.device)
        for block in self.blocks:
            h, x = block(h, x, flat_mask, covalent_adjacency)
        delta = (x - x0) * flat_mask[..., None]
        norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        delta = delta * torch.clamp(self.max_displacement / norm, max=1.0)
        return (x0 + delta).reshape(B, L, A, 3) * atom_mask[..., None]
