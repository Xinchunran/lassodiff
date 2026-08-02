"""Numerically stable latent candidate marginal objective."""
from __future__ import annotations

import torch


def candidate_marginal_loss(candidate_loss, prior, candidate_mask, temperature: float = 1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if candidate_loss.shape != prior.shape or prior.shape != candidate_mask.shape:
        raise ValueError("candidate loss/prior/mask must share shape [B,M]")
    valid = candidate_mask.bool()
    if not bool(valid.any(dim=1).all()):
        raise ValueError("each sample needs a valid candidate")
    safe_prior = torch.where(valid, prior, torch.ones_like(prior))
    if bool((safe_prior[valid] <= 0).any()):
        raise ValueError("valid candidate priors must be positive")
    normalized_prior = safe_prior * valid
    normalized_prior = normalized_prior / normalized_prior.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(prior.dtype).tiny)
    logits = torch.where(
        valid, normalized_prior.clamp(min=torch.finfo(prior.dtype).tiny).log() - candidate_loss / temperature,
        torch.full_like(candidate_loss, -torch.inf),
    )
    marginal = -temperature * torch.logsumexp(logits, dim=-1)
    posterior = torch.softmax(logits, dim=-1)
    posterior = torch.where(valid, posterior, torch.zeros_like(posterior))
    return marginal.mean(), posterior, normalized_prior
