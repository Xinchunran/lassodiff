from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class PairInitConfig:
    c_z: int = 128
    max_rel: int = 64
    edge_types: int = 5


class PairInitializer(nn.Module):
    def __init__(self, c_s_inputs: int, cfg: PairInitConfig = PairInitConfig()):
        super().__init__()
        self.cfg = cfg
        self.rel_emb = nn.Embedding(2 * cfg.max_rel + 1, cfg.c_z)
        self.s_to_z_i = nn.Linear(c_s_inputs, cfg.c_z, bias=False)
        self.s_to_z_j = nn.Linear(c_s_inputs, cfg.c_z, bias=False)
        self.edge_emb = nn.Embedding(cfg.edge_types, cfg.c_z)

    def forward(self, s_inputs, token_mask, k_ring_end: torch.Tensor, p_plug: torch.Tensor):
        B, L, _ = s_inputs.shape
        device = s_inputs.device

        pair_mask = token_mask[:, :, None] & token_mask[:, None, :]

        idx = torch.arange(L, device=device)
        rel = (idx[None, :, None] - idx[None, None, :]).clamp(-self.cfg.max_rel, self.cfg.max_rel)
        rel = rel + self.cfg.max_rel
        z = self.rel_emb(rel)
        z = z.repeat(B, 1, 1, 1)

        zi = self.s_to_z_i(s_inputs)
        zj = self.s_to_z_j(s_inputs)
        z = z + zi[:, :, None, :] + zj[:, None, :, :]

        edge_type = torch.zeros((B, L, L), dtype=torch.long, device=device)

        for d in [1, -1]:
            ii = torch.arange(L, device=device)
            jj = ii + d
            valid = (jj >= 0) & (jj < L)
            edge_type[:, ii[valid], jj[valid]] = 1

        k = k_ring_end.clamp(0, L - 1)
        b = torch.arange(B, device=device)
        edge_type[b, 0, k] = 2
        edge_type[b, k, 0] = 2

        a1 = torch.ones(B, dtype=torch.long, device=device) * 1
        a2 = (k // 2).clamp(0, L - 1)
        a3 = (k - 1).clamp(0, L - 1)
        p = p_plug.clamp(0, L - 1)
        for a in [a1, a2, a3]:
            edge_type[b, p, a] = 3
            edge_type[b, a, p] = 3

        edge_type[b, L - 1, a2] = 4
        edge_type[b, a2, L - 1] = 4

        z = z + self.edge_emb(edge_type)
        z = z * pair_mask.unsqueeze(-1).float()
        return z, pair_mask
