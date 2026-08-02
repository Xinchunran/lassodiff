"""Calibration and selective-classification metrics for the V3 sequence gate."""
from __future__ import annotations

import torch


def _binary_auc(probability, label):
    order = probability.argsort(descending=True)
    y = label[order].float()
    positives, negatives = y.sum(), (1 - y).sum()
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC/AUPRC require positive and negative examples")
    tpr = torch.cat([torch.zeros(1, device=y.device), y.cumsum(0) / positives])
    fpr = torch.cat([torch.zeros(1, device=y.device), (1 - y).cumsum(0) / negatives])
    auroc = torch.trapz(tpr, fpr)
    precision = y.cumsum(0) / torch.arange(1, len(y) + 1, device=y.device)
    auprc = (precision * y).sum() / positives
    return float(auroc), float(auprc)


def calibrate_gate_thresholds(probability, label, *, target_fpr, target_recall):
    probability, label = probability.float(), label.bool()
    if not 0 <= target_fpr < 1 or not 0 < target_recall <= 1:
        raise ValueError("invalid sequence calibration targets")
    negative, positive = probability[~label].sort(descending=True).values, probability[label].sort().values
    if not len(negative) or not len(positive):
        raise ValueError("calibration needs held-out positives and negatives")
    allowed = int(target_fpr * len(negative))
    accept = float(negative[allowed]) + torch.finfo(probability.dtype).eps if allowed < len(negative) else 1.0
    reject_index = min(len(positive) - 1, max(0, int((1 - target_recall) * len(positive))))
    reject = float(positive[reject_index])
    if reject >= accept:
        reject = max(0.0, accept - 1e-4)
    return {"reject_threshold": reject, "accept_threshold": min(1.0, accept)}


def sequence_classification_metrics(probability, label, decisions, *, bins=15, ood_mask=None):
    probability, label = probability.float(), label.bool()
    if probability.ndim != 1 or label.shape != probability.shape or len(decisions) != len(label):
        raise ValueError("sequence metrics inputs must have the same one-dimensional length")
    if not torch.isfinite(probability).all() or bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("sequence probability must be finite in [0,1]")
    auroc, auprc = _binary_auc(probability, label)
    predicted_positive = torch.tensor([item == "LASSO_PLAUSIBLE" for item in decisions], device=label.device)
    predicted_negative = torch.tensor([item == "NON_LASSO" for item in decisions], device=label.device)
    covered = predicted_positive | predicted_negative
    correct = (predicted_positive & label) | (predicted_negative & ~label)
    negative_count, positive_count = (~label).sum().clamp_min(1), label.sum().clamp_min(1)
    ece = probability.new_zeros(())
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (probability >= lower) & (probability < upper if index + 1 < bins else probability <= upper)
        if selected.any():
            ece += selected.float().mean() * (probability[selected].mean() - label[selected].float().mean()).abs()
    result = {
        "auroc": auroc, "auprc": auprc,
        "fpr": float((predicted_positive & ~label).sum() / negative_count),
        "recall": float((predicted_positive & label).sum() / positive_count),
        "specificity": float((predicted_negative & ~label).sum() / negative_count),
        "coverage": float(covered.float().mean()), "abstain_rate": float((~covered).float().mean()),
        "covered_accuracy": float(correct.sum() / covered.sum().clamp_min(1)),
        "ece": float(ece), "brier": float((probability - label.float()).square().mean()),
        "sample_count": len(label),
    }
    if ood_mask is not None:
        ood_mask = ood_mask.bool()
        result["ood_false_acceptance"] = float((predicted_positive & ood_mask).sum() / ood_mask.sum().clamp_min(1))
    return result
