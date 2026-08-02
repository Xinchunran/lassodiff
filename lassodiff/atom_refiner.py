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


def select_refiner_neighbors(coordinates: torch.Tensor, mask: torch.Tensor,
                             covalent_adjacency: torch.Tensor, max_neighbors: int) -> torch.Tensor:
    """Select all covalent neighbors first, then nearest valid non-covalent atoms."""
    if coordinates.ndim != 3 or mask.ndim != 2 or covalent_adjacency.ndim != 3:
        raise ValueError("neighbor inputs must be [B,N,3], [B,N], [B,N,N]")
    B, N = mask.shape
    K = min(int(max_neighbors), max(N - 1, 1))
    distances = torch.cdist(coordinates.float(), coordinates.float()).masked_fill(~mask[:, :, None] | ~mask[:, None, :], torch.inf)
    distances = distances.masked_fill(torch.eye(N, dtype=torch.bool, device=coordinates.device)[None], torch.inf)
    result = torch.empty((B, N, K), dtype=torch.long, device=coordinates.device)
    for batch in range(B):
        for node in range(N):
            bonded = torch.where(covalent_adjacency[batch, node] & mask[batch])[0]
            bonded = bonded[bonded != node]
            remaining = torch.argsort(distances[batch, node])
            selected = []
            for value in bonded.tolist() + remaining.tolist():
                if value != node and value not in selected:
                    selected.append(value)
                if len(selected) == K:
                    break
            if not selected:
                selected = [node] * K
            selected += [selected[-1]] * (K - len(selected))
            result[batch, node] = torch.tensor(selected[:K], dtype=torch.long, device=coordinates.device)
    return result


class MiniAtomRefinerV2(nn.Module):
    def __init__(self, hidden_dim: int = 96, layers: int = 4, max_neighbors: int = 32, max_displacement: float = .75):
        super().__init__()
        self.residue_embedding = nn.Embedding(21, hidden_dim)
        self.atom_embedding = nn.Embedding(14, hidden_dim)
        self.bond_embedding = nn.Embedding(8, hidden_dim)
        self.layers = int(layers)
        self.max_neighbors = int(max_neighbors)
        self.max_displacement = float(max_displacement)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers))

    def forward(self, coordinates, aa_ids, atom_mask, *, covalent_adjacency=None, bond_type=None):
        if coordinates.ndim != 4 or coordinates.shape[-2:] != (14, 3):
            raise ValueError("V2 refiner coordinates must be [B,L,14,3]")
        if self.training and (covalent_adjacency is None or bond_type is None):
            raise ValueError("V2 training refiner requires a covalent graph")
        B, L, A, _ = coordinates.shape
        if aa_ids.shape != (B, L) or atom_mask.shape != (B, L, A):
            raise ValueError("V2 refiner shape mismatch")
        if covalent_adjacency is None:
            covalent_adjacency = torch.zeros((B, L * A, L * A), dtype=torch.bool, device=coordinates.device)
        if bond_type is None:
            bond_type = covalent_adjacency.long()
        flat_mask = atom_mask.reshape(B, L * A)
        x0 = coordinates.reshape(B, L * A, 3)
        atom_ids = torch.arange(A, device=coordinates.device)[None, None].expand(B, L, A)
        h = (self.residue_embedding(aa_ids)[:, :, None] + self.atom_embedding(atom_ids)).reshape(B, L * A, -1) * flat_mask[..., None]
        for block in self.blocks:
            neighbors = select_refiner_neighbors(x0, flat_mask, covalent_adjacency, self.max_neighbors)
            batch = torch.arange(B, device=x0.device)[:, None, None]
            h_j = h[batch, neighbors]
            rel = x0[:, :, None] - x0[batch, neighbors]
            valid = flat_mask[:, :, None] & flat_mask[batch, neighbors]
            distance = rel.norm(dim=-1, keepdim=True).clamp_max(100)
            message = block(torch.cat((h[:, :, None].expand_as(h_j), h_j, distance), -1)) * valid[..., None]
            delta = (rel * message.mean(-1, keepdim=True)).sum(2) / valid.sum(2, keepdim=True).clamp_min(1)
            h = (h + message.sum(2) / valid.sum(2, keepdim=True).clamp_min(1)) * flat_mask[..., None]
            x0 = x0 + delta * flat_mask[..., None]
        delta = x0 - coordinates.reshape(B, L * A, 3)
        scale = torch.clamp(self.max_displacement / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8), max=1.0)
        return (coordinates.reshape(B, L * A, 3) + delta * scale).reshape(B, L, A, 3) * atom_mask[..., None]
