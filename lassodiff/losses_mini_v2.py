"""Chemistry-aware losses for torsion and Atom14 endpoints."""
from __future__ import annotations

import torch


def circular_velocity_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape or mask.shape != predicted.shape:
        raise ValueError("circular loss inputs must have equal shapes")
    delta = torch.atan2(torch.sin(predicted - target), torch.cos(predicted - target))
    valid = mask.to(delta.dtype)
    return (delta.square() * valid).sum() / valid.sum().clamp_min(1)


def softmin_conformer_loss(loss_per_conformer: torch.Tensor, conformer_mask: torch.Tensor, tau: float = .25) -> torch.Tensor:
    masked = loss_per_conformer.masked_fill(~conformer_mask, torch.inf)
    return (-tau * torch.logsumexp(-masked / tau, dim=-1)).mean()


_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20}


def all_atom_clash_loss(coordinates: torch.Tensor, mask: torch.Tensor, elements, *, bonded_12: torch.Tensor,
                        bonded_13: torch.Tensor, bonded_14: torch.Tensor) -> torch.Tensor:
    if coordinates.ndim != 3 or mask.ndim != 2:
        raise ValueError("clash inputs must be [B,N,3] and [B,N]")
    B, N = mask.shape
    distance = torch.cdist(coordinates.float(), coordinates.float())
    pair = mask[:, :, None] & mask[:, None, :]
    pair &= torch.triu(torch.ones((N, N), dtype=torch.bool, device=coordinates.device), diagonal=1)[None]
    pair &= ~bonded_12 & ~bonded_13
    pair &= ~bonded_14  # 1-4 contacts are conservatively excluded from severe clash loss.
    radius_values = torch.tensor([
        [[(_RADII.get(elements[b][i], 1.7) + _RADII.get(elements[b][j], 1.7)) * .70 for j in range(N)] for i in range(N)]
        for b in range(B)
    ], device=coordinates.device, dtype=coordinates.dtype)
    penalty = torch.relu(radius_values - distance).square() * pair
    return penalty.sum() / pair.sum().clamp_min(1)
