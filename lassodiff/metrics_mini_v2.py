"""Rollout metrics for paired Mini V2 evaluation."""
from __future__ import annotations

import torch


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
    if predicted_ca.shape != target_ca.shape:
        raise ValueError("lDDT inputs must have equal shape")
    pd = torch.cdist(predicted_ca[None], predicted_ca[None])[0]
    td = torch.cdist(target_ca[None], target_ca[None])[0]
    pair = ~torch.eye(pd.shape[0], dtype=torch.bool, device=pd.device)
    pair &= td < cutoff
    if not pair.any():
        return predicted_ca.new_tensor(0.0)
    error = (pd - td).abs()
    return ((error < .5) & pair).float().sum() / pair.float().sum().clamp_min(1)


def conformer_precision_coverage(generated_ca, target_ca, threshold=2.0):
    distances = torch.cdist(generated_ca, target_ca).mean(-1)
    precision = (distances.min(-1).values < threshold).float().mean()
    coverage = (distances.min(0).values < threshold).float().mean()
    return {"precision": float(precision), "coverage": float(coverage)}
