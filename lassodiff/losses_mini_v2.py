"""Chemistry-aware losses for torsion and Atom14 endpoints."""
from __future__ import annotations

import torch
from dataclasses import dataclass

from .atom_schema_lasso import ATOM_CB, ATOM_CISO, ATOM_N, ATOM_OISO
from .chi_geometry import _build_acceptor_reactive_group
from .residue_constants_mini import SYMMETRIC_ATOM_PAIRS, padded_atom14_names


def circular_velocity_loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked tangent-space velocity loss.

    Despite the historical name, velocity is not a periodic state.  Periodic
    wrapping belongs in state interpolation and endpoint angle comparisons,
    not in the velocity residual.
    """
    if predicted.shape != target.shape or mask.shape != predicted.shape:
        raise ValueError("circular loss inputs must have equal shapes")
    error = torch.nn.functional.smooth_l1_loss(predicted, target, reduction="none")
    valid = mask.to(error.dtype)
    return (error * valid).sum() / valid.sum().clamp_min(1)


def masked_tangent_velocity_loss(predicted: torch.Tensor, target: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
    return circular_velocity_loss(predicted, target, mask)


def circular_angle_loss(predicted: torch.Tensor, target: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    if predicted.shape != target.shape or mask.shape != predicted.shape:
        raise ValueError("circular angle loss inputs must have equal shapes")
    difference = torch.atan2(torch.sin(predicted - target), torch.cos(predicted - target))
    loss = torch.nn.functional.smooth_l1_loss(difference, torch.zeros_like(difference), reduction="none")
    weight = mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1)


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
    if predicted_core.ndim != 5 or target_core.ndim != 5 or target_mask.ndim != 4:
        raise ValueError("core tensors must be [B,Ns,L,A,3], [B,M,L,A,3], [B,M,L,A]")
    difference = predicted_core[:, :, None] - target_core[:, None]
    squared_distance = difference.square().sum(dim=-1)
    mask = target_mask[:, None].to(squared_distance.dtype)
    per = (squared_distance * mask).sum(dim=(-1, -2)) / mask.sum(dim=(-1, -2)).clamp_min(1)
    valid = conformer_mask[:, None].bool()
    logits = (-per / tau).masked_fill(~valid, float("-inf"))
    count = valid.sum(dim=-1).clamp_min(1).to(logits.dtype)
    return (-tau * (torch.logsumexp(logits, dim=-1) - count.log())).mean()


def atom14_local_loss(predicted, target, target_mask):
    """Masked Atom14 coordinate loss in the canonical decoder frame."""
    if predicted.shape != target.shape or target_mask.shape != predicted.shape[:-1]:
        raise ValueError("Atom14 local-loss shapes do not match")
    error = (predicted - target).square().sum(dim=-1)
    weight = target_mask.to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def symmetry_aware_atom14_loss(predicted, target, target_mask, sequences,
                               *, acceptor_indices=None):
    """Atom14 loss minimized over chemically equivalent target swaps."""
    if predicted.ndim != 4:
        raise ValueError("predicted Atom14 must be [B,L,14,3]")
    total = predicted.new_zeros(())
    for b, sequence in enumerate(sequences):
        names = padded_atom14_names(sequence, None)
        best = predicted.new_full((len(sequence),), float("inf"))
        for residue, aa in enumerate(sequence):
            candidates = [target[b, residue]]
            for left, right in SYMMETRIC_ATOM_PAIRS.get(aa, ()):
                swapped = target[b, residue].clone()
                lookup = {name: slot for slot, name in enumerate(names[residue]) if name}
                if left in lookup and right in lookup:
                    swapped[[lookup[left], lookup[right]]] = swapped[[lookup[right], lookup[left]]]
                candidates.append(swapped)
            values = []
            for candidate_target in candidates:
                values.append(atom14_local_loss(
                    predicted[b, residue:residue + 1],
                    candidate_target[None],
                    target_mask[b, residue:residue + 1],
                ))
            best[residue] = torch.stack(values).min()
        total = total + best.mean()
    return total / max(len(sequences), 1)


def scheduled_weight(step: int, *, start_step: int, ramp_steps: int, final_weight: float) -> float:
    if step < start_step:
        return 0.0
    return final_weight * min(1.0, (step - start_step + 1) / max(ramp_steps, 1))


def iso_geometry_loss(core, candidates, token_mask, acceptor_chi=None, acceptor_chi_mask=None):
    values = []
    for b, candidate in enumerate(candidates):
        k = candidate.k
        for sample in range(core.shape[1]):
            n0 = core[b, sample, 0, ATOM_N]
            ciso = core[b, sample, k, ATOM_CISO]
            oiso = core[b, sample, k, ATOM_OISO]
            if candidate.sequence[k] == "D":
                predecessor = core[b, sample, k, ATOM_CB]
            elif acceptor_chi is not None and acceptor_chi_mask is not None:
                cg, _ciso, _oiso = _build_acceptor_reactive_group(
                    sequence=candidate.sequence, core=core[b, sample, :len(candidate.sequence)],
                    acceptor_index=k, chi=acceptor_chi[b, sample, k],
                    chi_mask=acceptor_chi_mask[b, sample, k],
                )
                predecessor = cg
            else:
                predecessor = core[b, sample, k, ATOM_CB]
            oxygen = oiso - ciso
            side = predecessor - ciso
            closure = n0 - ciso
            oxygen_cos = (oxygen * closure).sum() / (oxygen.norm() * closure.norm()).clamp_min(1e-8)
            side_cos = (side * closure).sum() / (side.norm() * closure.norm()).clamp_min(1e-8)
            target_cos = core.new_tensor(-.5)
            normal = torch.linalg.cross(oxygen, side, dim=-1)
            plane = torch.dot(closure, normal).abs() / normal.norm().clamp_min(1e-8)
            values.extend((
                ((closure.norm() - 1.33) / .25).square(),
                ((oxygen_cos - target_cos) / .35).square(),
                ((side_cos - target_cos) / .35).square(),
                (plane / .50).square(),
                ((oxygen.norm() - 1.24) / .10).square(),
                ((side.norm() - 1.522) / .15).square(),
            ))
    return torch.stack(values).mean() if values else core.new_zeros(())


def core_clash_surrogate(core, candidates, token_mask):
    """Full-chain core clash surrogate with covalent exclusions."""
    if core.ndim != 5:
        raise ValueError("core must be [B,Ns,L,A,3]")
    B, Ns, L, A, _ = core.shape
    flat = core.reshape(B * Ns, L * A, 3)
    valid_rows = torch.zeros((B, L, A), dtype=torch.bool, device=core.device)
    for b, candidate in enumerate(candidates):
        valid_rows[b, :len(candidate.sequence)] = candidate.core_atom_mask.to(core.device)
    valid = valid_rows[:, None].expand(B, Ns, L, A).reshape(B * Ns, L * A)
    exclusion = torch.zeros((B, L * A, L * A), dtype=torch.bool, device=core.device)
    for b, candidate in enumerate(candidates):
        def connect(ri, ai, rj, aj):
            i, j = ri * A + ai, rj * A + aj
            exclusion[b, i, j] = exclusion[b, j, i] = True
        for r in range(len(candidate.sequence)):
            for ai, aj in ((0, 1), (1, 2), (2, 3)):
                connect(r, ai, r, aj)
            if candidate.sequence[r] != "G":
                connect(r, 1, r, 4)
            if r + 1 < len(candidate.sequence):
                connect(r, 2, r + 1, 0)
        connect(candidate.k, 4, candidate.k, 5)
        connect(candidate.k, 5, candidate.k, 6)
        connect(0, 0, candidate.k, 5)
    adjacency = exclusion.float()
    bonded13 = torch.bmm(adjacency, adjacency).gt(0) & ~exclusion
    bonded14 = torch.bmm(bonded13.float(), adjacency).gt(0) & ~exclusion & ~bonded13
    exclusion = exclusion | bonded13 | bonded14
    exclusion = exclusion.expand(B * Ns, -1, -1)
    pair = valid[:, :, None] & valid[:, None, :]
    pair &= ~exclusion
    pair &= torch.triu(torch.ones_like(pair), diagonal=1)
    distances = torch.cdist(flat.float(), flat.float())
    violations = torch.relu(1.8 - distances)
    return (violations.square() * pair).sum() / pair.sum().clamp_min(1)


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
