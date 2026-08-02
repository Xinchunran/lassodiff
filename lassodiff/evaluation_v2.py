"""Target-relative structure and lasso-topology metrics for V2 validation.

The MD/template labels are candidate specific: candidate rank 1 is evaluated
against min1/relax1, and so on.  Metrics in this module therefore preserve the
candidate dimension and never compare every candidate to one copied target.
"""
from __future__ import annotations

from typing import Dict

import torch

from .losses_lasso import ATOM_C, ATOM_CA, ATOM_CISO, ATOM_N, ATOM_O, gauss_linking_integral_ca
from .topology_adapter import CandidateBatch


def masked_aligned_rmsd(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Kabsch-aligned RMSD per ``[B,M]`` candidate, over all remaining points."""
    if pred.shape != target.shape or pred.shape[-1] != 3:
        raise ValueError("pred and target must have identical [...,3] shape")
    if pred.ndim < 4 or pred.shape[:2] != mask.shape[:2] or pred.shape[2:-1] != mask.shape[2:]:
        raise ValueError("mask must match pred without the final coordinate axis")
    B, M = pred.shape[:2]
    p = pred.float().reshape(B, M, -1, 3)
    q = target.float().reshape(B, M, -1, 3)
    w = mask.bool().reshape(B, M, -1).float()
    count = w.sum(dim=-1).clamp(min=1.0)
    p_center = (p * w[..., None]).sum(dim=-2) / count[..., None]
    q_center = (q * w[..., None]).sum(dim=-2) / count[..., None]
    p0 = (p - p_center[..., None, :]) * w[..., None]
    q0 = (q - q_center[..., None, :]) * w[..., None]
    covariance = torch.einsum("bmni,bmnj->bmij", p0, q0)
    u, _s, vh = torch.linalg.svd(covariance)
    sign = torch.where(
        torch.linalg.det(torch.matmul(u, vh)) < 0,
        -torch.ones((B, M), device=pred.device),
        torch.ones((B, M), device=pred.device),
    )
    correction = torch.zeros((B, M, 3, 3), device=pred.device, dtype=torch.float32)
    correction[..., 0, 0] = 1.0
    correction[..., 1, 1] = 1.0
    correction[..., 2, 2] = sign
    rotation = torch.matmul(torch.matmul(u, correction), vh)
    aligned = torch.matmul(p0, rotation)
    squared = ((aligned - q0).square().sum(dim=-1) * w).sum(dim=-1) / count
    return torch.sqrt(squared.clamp(min=0.0))


def _candidate_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    weight = valid.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp(min=1.0)


def _linking_number(coords: torch.Tensor, token_mask: torch.Tensor, candidates: CandidateBatch) -> torch.Tensor:
    B, M, L, _A, _C = coords.shape
    ca = coords[..., ATOM_CA, :].reshape(B * M, 1, L, 3)
    tokens = token_mask[:, None].expand(B, M, L).reshape(B * M, L)
    return gauss_linking_integral_ca(
        ca, candidates.k.reshape(-1), candidates.p.reshape(-1), tokens
    ).reshape(B, M)


def _link_class(value: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    return torch.where(value > threshold, 1, torch.where(value < -threshold, -1, 0))


def _iso_distance(coords: torch.Tensor, candidates: CandidateBatch) -> torch.Tensor:
    B, M, L, _A, _C = coords.shape
    b = torch.arange(B, device=coords.device)[:, None].expand(B, M)
    m = torch.arange(M, device=coords.device)[None, :].expand(B, M)
    acceptor = candidates.acceptor_index.clamp(0, L - 1)
    nterm = coords[:, :, 0, ATOM_N]
    ciso = coords[b, m, acceptor, ATOM_CISO]
    return torch.linalg.vector_norm(nterm - ciso, dim=-1)


def _bond_length_mae(coords: torch.Tensor, token_mask: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
    valid_token = token_mask[:, None].expand(coords.shape[:3])
    terms = []
    weights = []

    def add(res_a, atom_a, res_b, atom_b, expected, token_pair):
        valid = token_pair & atom_mask[..., res_a, atom_a] & atom_mask[..., res_b, atom_b]
        distance = torch.linalg.vector_norm(
            coords[..., res_a, atom_a, :] - coords[..., res_b, atom_b, :], dim=-1
        )
        terms.append((distance - expected).abs() * valid)
        weights.append(valid)

    L = coords.shape[2]
    idx = torch.arange(L, device=coords.device)
    add(idx, ATOM_N, idx, ATOM_CA, 1.46, valid_token)
    add(idx, ATOM_CA, idx, ATOM_C, 1.52, valid_token)
    add(idx, ATOM_C, idx, ATOM_O, 1.23, valid_token)
    if L > 1:
        pair_valid = valid_token[..., :-1] & valid_token[..., 1:]
        add(idx[:-1], ATOM_C, idx[1:], ATOM_N, 1.33, pair_valid)
    numerator = sum(term.sum() for term in terms)
    denominator = sum(weight.sum() for weight in weights).clamp(min=1)
    return numerator / denominator


def _ca_lddt(pred: torch.Tensor, target: torch.Tensor, token_mask: torch.Tensor, valid_candidates: torch.Tensor) -> torch.Tensor:
    pred_ca, target_ca = pred[..., ATOM_CA, :].float(), target[..., ATOM_CA, :].float()
    d_pred = torch.cdist(pred_ca, pred_ca)
    d_target = torch.cdist(target_ca, target_ca)
    L = pred.shape[2]
    pair = token_mask[:, None, :, None] & token_mask[:, None, None, :]
    pair = pair & ~torch.eye(L, dtype=torch.bool, device=pred.device)[None, None]
    pair = pair & (d_target < 15.0) & valid_candidates[:, :, None, None]
    delta = (d_pred - d_target).abs()
    score = sum((delta < cutoff).float() for cutoff in (0.5, 1.0, 2.0, 4.0)) / 4.0
    return (score * pair).sum() / pair.sum().clamp(min=1)


@torch.no_grad()
def evaluate_generated_structures(
    predicted: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    candidates: CandidateBatch,
) -> Dict[str, float]:
    """Compare sampled candidate structures with their matching MD templates.

    ``topology_match_rate`` requires both the native Gauss-link class and the
    native isopeptide distance (within 0.5 Å).  It is deliberately
    target-relative, because the MD/template set contains structures whose
    absolute closure geometry is not uniformly ideal.
    """
    if predicted.shape != target.shape or predicted.ndim != 5:
        raise ValueError("predicted/target must be [B,M,L,A,3]")
    if atom_mask.shape != predicted.shape[:-1]:
        raise ValueError("candidate atom_mask must be [B,M,L,A]")
    valid = candidates.candidate_mask.bool()
    ca_mask = atom_mask[..., ATOM_CA] & token_mask[:, None] & valid[:, :, None]
    bb_mask = atom_mask[..., :4] & token_mask[:, None, :, None] & valid[:, :, None, None]
    ca_rmsd = masked_aligned_rmsd(predicted[..., ATOM_CA, :], target[..., ATOM_CA, :], ca_mask)
    bb_rmsd = masked_aligned_rmsd(predicted[..., :4, :], target[..., :4, :], bb_mask)
    pred_iso, target_iso = _iso_distance(predicted, candidates), _iso_distance(target, candidates)
    iso_error = (pred_iso - target_iso).abs()
    pred_link = _linking_number(predicted, token_mask, candidates)
    target_link = _linking_number(target, token_mask, candidates)
    link_match = _link_class(pred_link).eq(_link_class(target_link))
    topology_match = link_match & (iso_error <= 0.5)
    candidate_count = int(valid.sum().item())
    return {
        "candidate_count": candidate_count,
        "ca_rmsd": float(_candidate_mean(ca_rmsd, valid).item()),
        "backbone_rmsd": float(_candidate_mean(bb_rmsd, valid).item()),
        "ca_lddt": float(_ca_lddt(predicted, target, token_mask, valid).item()),
        "iso_distance_mae": float(_candidate_mean(iso_error, valid).item()),
        "link_number_mae": float(_candidate_mean((pred_link - target_link).abs(), valid).item()),
        "link_class_accuracy": float(_candidate_mean(link_match.float(), valid).item()),
        "topology_match_rate": float(_candidate_mean(topology_match.float(), valid).item()),
        "bond_length_mae": float(_bond_length_mae(predicted, token_mask, atom_mask).item()),
    }
