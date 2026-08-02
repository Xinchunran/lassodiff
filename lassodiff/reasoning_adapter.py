from __future__ import annotations

import torch
import torch.nn as nn


class OpenDDEReasoningAdapter(nn.Module):
    def __init__(self, c_s_open: int, c_z_open: int, c_s: int, c_z: int):
        super().__init__()
        self.single = nn.Sequential(nn.LayerNorm(c_s_open), nn.Linear(c_s_open, c_s))
        self.pair = nn.Sequential(nn.LayerNorm(c_z_open), nn.Linear(c_z_open, c_z))

    def forward(self, single, pair, token_mask):
        pair_mask = token_mask[:, :, None] & token_mask[:, None, :]
        adapted_single = self.single(single) * token_mask[..., None].to(single.dtype)
        adapted_pair = self.pair(pair) * pair_mask[..., None].to(pair.dtype)
        return adapted_single, adapted_pair


def scale_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    return value.detach() + float(scale) * (value - value.detach())
