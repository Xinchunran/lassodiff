from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

ADAPTED_FROM = "adapted from SimpleFold"


@dataclass
class SimpleFoldCfg:
    c_s: int = 384
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1


class SimpleFoldConditioner(nn.Module):
    def __init__(self, c_s_inputs: int, cfg: SimpleFoldCfg = SimpleFoldCfg()):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(c_s_inputs, cfg.c_s)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.c_s,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.c_s * cfg.ff_mult,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        self.t_mlp = nn.Sequential(
            nn.Linear(1, cfg.c_s),
            nn.SiLU(),
            nn.Linear(cfg.c_s, cfg.c_s),
        )

    def forward(self, s_inputs, z_init, token_mask, t_scalar: torch.Tensor):
        x = self.in_proj(s_inputs)

        if t_scalar.dim() == 2:
            t_feat = t_scalar.mean(dim=1, keepdim=True)
        else:
            t_feat = t_scalar.unsqueeze(-1)
        x = x + self.t_mlp(t_feat).unsqueeze(1)

        key_padding_mask = ~token_mask
        s_trunk = self.trunk(x, src_key_padding_mask=key_padding_mask)
        return s_trunk, z_init
