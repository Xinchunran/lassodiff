"""Rollout metrics for paired Mini V2 evaluation."""
from __future__ import annotations

import torch


def aligned_ca_rmsd(predicted_ca: torch.Tensor, target_ca: torch.Tensor) -> torch.Tensor:
    """Kabsch RMSD for diagnostics; strict validity never uses alignment."""
    if predicted_ca.shape != target_ca.shape or predicted_ca.ndim != 2 or predicted_ca.shape[-1] != 3:
        raise ValueError("aligned CA RMSD inputs must both be [L,3]")
    pred_center = predicted_ca.mean(0)
    target_center = target_ca.mean(0)
    pred = predicted_ca - pred_center
    target = target_ca - target_center
    covariance = pred.T @ target
    u, _, vh = torch.linalg.svd(covariance)
    sign = torch.det(u @ vh).sign()
    correction = torch.eye(3, device=pred.device, dtype=pred.dtype)
    correction[-1, -1] = sign
    rotation = u @ correction @ vh
    aligned = pred @ rotation + target_center
    return torch.sqrt((aligned - target_ca).square().sum(-1).mean())


def best_ca_rmsd(predicted, targets, target_mask=None):
    pred = predicted[..., 1, :] if predicted.shape[-2] >= 2 else predicted
    targ = targets[..., 1, :] if targets.shape[-2] >= 2 else targets
    if target_mask is not None:
        mask = target_mask[..., 1].bool()
    else:
        mask = torch.ones(targ.shape[:-1], dtype=torch.bool, device=targ.device)
    values = []
    for i in range(pred.shape[0]):
        diffs = pred[i, None] - targ
        denom = mask.sum(-1).clamp_min(1)
        values.append(torch.sqrt((diffs.square().sum(-1) * mask).sum(-1) / denom))
    return torch.stack(values).min()


def lddt_score(predicted_ca, target_ca, cutoff=15.0):
    """Compute the standard CA lDDT fraction over four local thresholds."""
    if predicted_ca.shape != target_ca.shape:
        raise ValueError("lDDT inputs must have equal shape")
    pd = torch.cdist(predicted_ca[None], predicted_ca[None])[0]
    td = torch.cdist(target_ca[None], target_ca[None])[0]
    pair = ~torch.eye(pd.shape[0], dtype=torch.bool, device=pd.device)
    pair &= td < cutoff
    if not pair.any():
        return predicted_ca.new_tensor(0.0)
    error = (pd - td).abs()
    thresholds = predicted_ca.new_tensor((.5, 1.0, 2.0, 4.0))
    local = (error[..., None] < thresholds).to(predicted_ca.dtype).mean(-1)
    return (local * pair).sum() / pair.to(predicted_ca.dtype).sum().clamp_min(1)


def conformer_precision_coverage(generated_ca, target_ca, threshold=2.0):
    distances = torch.cdist(generated_ca, target_ca).mean(-1)
    precision = (distances.min(-1).values < threshold).float().mean()
    coverage = (distances.min(0).values < threshold).float().mean()
    return {"precision": float(precision), "coverage": float(coverage)}
