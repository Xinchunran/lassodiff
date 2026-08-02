"""Chemistry-aware losses for torsion and Atom14 endpoints."""
from __future__ import annotations

import torch
from dataclasses import dataclass


def circular_velocity_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape or mask.shape != predicted.shape:
        raise ValueError("circular loss inputs must have equal shapes")
    delta = torch.atan2(torch.sin(predicted - target), torch.cos(predicted - target))
    valid = mask.to(delta.dtype)
    return (delta.square() * valid).sum() / valid.sum().clamp_min(1)


def softmin_conformer_loss(loss_per_conformer: torch.Tensor, conformer_mask: torch.Tensor, tau: float = .25) -> torch.Tensor:
    masked = loss_per_conformer.masked_fill(~conformer_mask, torch.inf)
    return (-tau * torch.logsumexp(-masked / tau, dim=-1)).mean()


@dataclass
class MiniV2LossOutput:
    total: torch.Tensor
    backbone_flow: torch.Tensor
    acceptor_chi_flow: torch.Tensor
    endpoint_core: torch.Tensor
    endpoint_pair: torch.Tensor
    iso_geometry: torch.Tensor
    topology: torch.Tensor
    clash: torch.Tensor
    sidechain: torch.Tensor
    refine: torch.Tensor
    viability: torch.Tensor


def conformer_softmin_core_loss(predicted_core, target_core, target_mask, conformer_mask, tau=.25):
    difference = predicted_core[:, :, None] - target_core[:, None]
    mask = target_mask[:, None, :, :, :, None].to(difference.dtype)
    per = (difference.square() * mask).sum(dim=(-1, -2, -3, -4)) / mask.sum(dim=(-1, -2, -3, -4)).clamp_min(1)
    per = per.masked_fill(~conformer_mask[:, None], torch.inf)
    return (-tau * torch.logsumexp(-per / tau, dim=-1)).mean()


def scheduled_weight(step: int, *, start_step: int, ramp_steps: int, final_weight: float) -> float:
    if step < start_step:
        return 0.0
    return final_weight * min(1.0, (step - start_step + 1) / max(ramp_steps, 1))


def iso_geometry_loss(core, candidates, token_mask):
    values = []
    for b, candidate in enumerate(candidates):
        k = candidate.k
        n0, ciso, oiso = core[b, :, 0, 0], core[b, :, k, 5], core[b, :, k, 6]
        # Normalize the Å-scale residuals so this surrogate remains a
        # well-conditioned auxiliary while its scheduled coefficient ramps.
        values.extend((((n0 - ciso).norm(dim=-1) - 1.33) / 4.0).square())
        values.extend((((ciso - oiso).norm(dim=-1) - 1.24) / 2.0).square())
    return torch.stack(values).mean() if values else core.new_zeros(())


def core_clash_surrogate(core, candidates, token_mask):
    # Core-7 excludes bonded pairs only approximately; the chemistry-aware
    # Atom14 clash path handles the full graph.  This term is finite and small
    # during Stage A without rewarding a topology seed.
    distance = torch.cdist(core[..., :4, :].reshape(-1, 4, 3), core[..., :4, :].reshape(-1, 4, 3))
    pair = torch.triu(torch.ones((4, 4), dtype=torch.bool, device=core.device), diagonal=1)
    return torch.relu(1.15 - distance)[..., pair].square().mean()


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
    radius_values = torch.tensor([
        [[(_RADII.get(elements[b][i], 1.7) + _RADII.get(elements[b][j], 1.7)) * .70 for j in range(N)] for i in range(N)]
        for b in range(B)
    ], device=coordinates.device, dtype=coordinates.dtype)
    weight = torch.where(bonded_14, torch.full_like(distance, .25), torch.ones_like(distance))
    penalty = torch.relu(radius_values - distance).square() * pair * weight
    return penalty.sum() / pair.sum().clamp_min(1)
