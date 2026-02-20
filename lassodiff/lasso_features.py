from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class LassoFeatConfig:
    clip: int = 64


def build_lasso_features(
    L: int,
    k_ring_end: int,
    p_plug: int,
    device: torch.device,
    cfg: LassoFeatConfig = LassoFeatConfig(),
) -> torch.Tensor:
    idx = torch.arange(L, device=device)

    is_ring = (idx <= k_ring_end).float()
    is_loop = ((idx > k_ring_end) & (idx <= p_plug)).float()
    is_tail = (idx > p_plug).float()

    dist_k = (idx - k_ring_end).clamp(-cfg.clip, cfg.clip) / cfg.clip
    dist_p = (idx - p_plug).clamp(-cfg.clip, cfg.clip) / cfg.clip

    is_N = torch.zeros(L, device=device)
    is_N[0] = 1.0
    is_C = torch.zeros(L, device=device)
    is_C[-1] = 1.0

    feats = torch.stack([is_ring, is_loop, is_tail, dist_k, dist_p, is_N, is_C], dim=-1)
    return feats
