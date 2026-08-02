from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SequenceAssessment:
    no_lasso_logit: torch.Tensor
    acceptor_logits: torch.Tensor
    plug_logits: torch.Tensor
    candidate_logits: torch.Tensor
    sequence_embedding: torch.Tensor
    ood_score: torch.Tensor


def _masked_mean(value, mask, dimensions):
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight[..., None]
    return (value * weight).sum(dim=dimensions) / weight.sum(dim=dimensions).clamp(min=1.0)


class LassoSequenceGate(nn.Module):
    """Candidate-independent H0/hypothesis head over OpenDDE residue reasoning."""

    def __init__(self, c_s: int, c_z: int, hidden_dim: int = 384, max_candidates: int = 3):
        super().__init__()
        self.single_pool_score = nn.Linear(c_s, 1)
        self.pair_pool_score = nn.Linear(c_z, 1)
        self.fuse = nn.Sequential(
            nn.LayerNorm(2 * c_s + 2 * c_z), nn.Linear(2 * c_s + 2 * c_z, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.no_lasso = nn.Linear(hidden_dim, 1)
        self.acceptor_head = nn.Linear(c_s, 1)
        self.plug_head = nn.Linear(c_s, 1)
        self.candidate_head = nn.Linear(hidden_dim, max_candidates)
        self.embedding_head = nn.Linear(hidden_dim, hidden_dim)
        self.ood_head = nn.Linear(hidden_dim, 1)

    def forward(self, s_res, z_res, token_mask, residue_type=None, residue_index=None) -> SequenceAssessment:
        if s_res.shape[:2] != token_mask.shape or z_res.shape[:3] != (
            token_mask.shape[0], token_mask.shape[1], token_mask.shape[1]
        ):
            raise ValueError("sequence gate reasoning shapes do not match token mask")
        single_scores = self.single_pool_score(s_res).squeeze(-1).masked_fill(~token_mask, -torch.inf)
        single_weights = torch.softmax(single_scores, dim=-1)
        single_global = (s_res * single_weights[..., None]).sum(dim=1)
        single_nterm = s_res[:, 0]

        nterm_pair_scores = self.pair_pool_score(z_res[:, 0]).squeeze(-1).masked_fill(~token_mask, -torch.inf)
        nterm_pair_weights = torch.softmax(nterm_pair_scores, dim=-1)
        pair_nterm = (z_res[:, 0] * nterm_pair_weights[..., None]).sum(dim=1)

        pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
        pair_scores = self.pair_pool_score(z_res).squeeze(-1).masked_fill(~pair_mask, -torch.inf)
        pair_weights = torch.softmax(pair_scores.flatten(1), dim=-1).reshape_as(pair_scores)
        pair_global = (z_res * pair_weights[..., None]).sum(dim=(1, 2))
        hidden = self.fuse(torch.cat([single_global, single_nterm, pair_nterm, pair_global], dim=-1))

        minimum = torch.finfo(s_res.dtype).min
        acceptor_logits = self.acceptor_head(s_res).squeeze(-1).masked_fill(~token_mask, minimum)
        plug_logits = self.plug_head(s_res).squeeze(-1).masked_fill(~token_mask, minimum)
        if residue_type is not None:
            acceptor_allowed = ((residue_type == 2) | (residue_type == 3)) & token_mask
            acceptor_logits = acceptor_logits.masked_fill(~acceptor_allowed, minimum)
        if residue_index is not None:
            plug_logits = plug_logits.masked_fill(~(token_mask & (residue_index > 0)), minimum)
        return SequenceAssessment(
            no_lasso_logit=self.no_lasso(hidden).squeeze(-1),
            acceptor_logits=acceptor_logits,
            plug_logits=plug_logits,
            candidate_logits=self.candidate_head(hidden),
            sequence_embedding=self.embedding_head(hidden),
            ood_score=torch.sigmoid(self.ood_head(hidden).squeeze(-1)),
        )


def decide_sequence(
    lasso_probability: torch.Tensor,
    ood_score: torch.Tensor,
    *,
    reject_threshold: float,
    accept_threshold: float,
    ood_threshold: float,
) -> list[str]:
    if not 0 <= reject_threshold < accept_threshold <= 1:
        raise ValueError("sequence decision thresholds are invalid")
    decisions = []
    for probability, ood in zip(lasso_probability.tolist(), ood_score.tolist()):
        if ood >= ood_threshold:
            decisions.append("ABSTAIN")
        elif probability <= reject_threshold:
            decisions.append("NON_LASSO")
        elif probability >= accept_threshold:
            decisions.append("LASSO_PLAUSIBLE")
        else:
            decisions.append("ABSTAIN")
    return decisions
