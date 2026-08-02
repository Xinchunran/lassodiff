"""Explicit candidate-specific topology features for OpenDDE LassoDiff V3."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class CandidateBatch:
    k: torch.Tensor                 # [B, M]
    p: torch.Tensor                 # [B, M]
    acceptor_index: torch.Tensor    # [B, M]
    prior: torch.Tensor             # [B, M]
    candidate_mask: torch.Tensor    # [B, M]
    acceptor_type: torch.Tensor | None = None  # [B,M], optional ASP/GLU class

    def validate(self, length: int) -> None:
        shapes = {tuple(value.shape) for value in (self.k, self.p, self.acceptor_index, self.prior, self.candidate_mask)}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError("candidate tensors must all have shape [B, M]")
        valid = self.candidate_mask.bool()
        if not bool(valid.any(dim=1).all()):
            raise ValueError("each batch item needs at least one valid candidate")
        for name, value in (("k", self.k), ("p", self.p), ("acceptor_index", self.acceptor_index)):
            if bool(((value[valid] < 0) | (value[valid] >= length)).any()):
                raise ValueError(f"candidate {name} outside sequence length")
        if bool((self.p[valid] <= self.k[valid]).any()):
            raise ValueError("candidate requires p > k")
        if bool((self.prior[valid] <= 0).any()):
            raise ValueError("candidate prior must be positive")
        if self.acceptor_type is not None and tuple(self.acceptor_type.shape) != tuple(self.k.shape):
            raise ValueError("candidate acceptor_type must have shape [B,M]")


class LassoTopologyAdapterV3(nn.Module):
    """Inject candidate topology into both single and pair reasoning states."""

    def __init__(self, c_s: int, c_z: int, n_features: int = 10):
        super().__init__()
        self.single_projection = nn.Sequential(
            nn.Linear(n_features, c_s), nn.GELU(), nn.Linear(c_s, c_s)
        )
        self.pair_projection = nn.Sequential(
            nn.Linear(n_features, c_z), nn.GELU(), nn.Linear(c_z, c_z)
        )

    @staticmethod
    def residue_features(candidates: CandidateBatch, length: int, dtype: torch.dtype):
        B, M = candidates.k.shape
        idx = torch.arange(length, device=candidates.k.device)[None, None].expand(B, M, -1)
        k, p, acceptor = candidates.k[..., None], candidates.p[..., None], candidates.acceptor_index[..., None]
        denominator = max(length - 1, 1)
        prior = candidates.prior[..., None].expand(B, M, length).to(dtype)
        acceptor_type = (
            torch.zeros_like(prior) if candidates.acceptor_type is None
            else candidates.acceptor_type[..., None].expand(B, M, length).to(dtype)
        )
        return torch.stack([
            (idx <= k).to(dtype), ((idx > k) & (idx <= p)).to(dtype), (idx > p).to(dtype),
            (idx == acceptor).to(dtype), (idx == p).to(dtype), (idx == 0).to(dtype),
            (idx - acceptor).abs().to(dtype) / denominator,
            (idx - p).abs().to(dtype) / denominator, prior, acceptor_type,
        ], dim=-1)

    def forward(self, s_base, z_base, token_mask, candidates: CandidateBatch):
        B, L, C = s_base.shape
        candidates.validate(L)
        features = self.residue_features(candidates, L, s_base.dtype)
        candidate_valid = candidates.candidate_mask[:, :, None]
        residue_mask = token_mask[:, None] & candidate_valid
        pair_mask = residue_mask[:, :, :, None] & residue_mask[:, :, None, :]
        single = s_base[:, None] + self.single_projection(features)
        pair_features = features[:, :, :, None] + features[:, :, None, :]
        pair = z_base[:, None] + self.pair_projection(pair_features)
        return (
            single * residue_mask[..., None].to(single.dtype),
            pair * pair_mask[..., None].to(pair.dtype),
            pair_mask,
        )
