"""Losses for candidate-independent H0 and candidate hypothesis heads."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_gate_objective(
    assessment, *, label, token_mask, acceptor_index, plug_index, teacher_prior,
    candidate_mask, ood_target, gate_weight=1.0, hypothesis_weight=1.0,
    position_weight=1.0, ood_weight=1.0,
):
    label = label.float()
    if label.ndim != 1 or assessment.no_lasso_logit.shape != label.shape:
        raise ValueError("sequence gate labels must be [B]")
    # H0=1 for NON_LASSO.
    gate = F.binary_cross_entropy_with_logits(assessment.no_lasso_logit, 1 - label)
    positives = label.bool()
    position_terms = []
    if positives.any():
        if bool((acceptor_index[positives] < 0).any()) or bool((plug_index[positives] < 0).any()):
            raise ValueError("positive sequence is missing acceptor/plug target")
        acceptor_logits = assessment.acceptor_logits[positives].masked_fill(
            ~token_mask[positives], torch.finfo(assessment.acceptor_logits.dtype).min
        )
        plug_logits = assessment.plug_logits[positives].masked_fill(
            ~token_mask[positives], torch.finfo(assessment.plug_logits.dtype).min
        )
        position_terms.extend([
            F.cross_entropy(acceptor_logits, acceptor_index[positives]),
            F.cross_entropy(plug_logits, plug_index[positives]),
        ])
    position = sum(position_terms) if position_terms else assessment.no_lasso_logit.sum() * 0
    valid_hypothesis = candidate_mask.bool() & positives[:, None]
    if valid_hypothesis.any():
        prior = torch.where(valid_hypothesis, teacher_prior, torch.zeros_like(teacher_prior))
        prior = prior / prior.sum(-1, keepdim=True).clamp_min(torch.finfo(prior.dtype).tiny)
        log_probability = torch.log_softmax(
            assessment.candidate_logits.masked_fill(~candidate_mask, -torch.inf), dim=-1
        )
        marginal = torch.logsumexp(
            torch.where(valid_hypothesis, prior.clamp_min(1e-12).log() + log_probability, -torch.inf), dim=-1
        )
        hypothesis = -marginal[positives].mean()
    else:
        hypothesis = assessment.no_lasso_logit.sum() * 0
    ood_probability = assessment.ood_score.clamp(1e-6, 1 - 1e-6)
    ood = F.binary_cross_entropy(ood_probability, ood_target.float())
    total = gate_weight * gate + hypothesis_weight * hypothesis + position_weight * position + ood_weight * ood
    return {"total": total, "gate": gate, "hypothesis": hypothesis, "position": position, "ood": ood}
