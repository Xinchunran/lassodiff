from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn

ADAPTED_FROM = "adapted from Protenix"


@dataclass
class PairformerCfg:
    n_blocks: int = 8
    n_heads: int = 8
    c_s: int = 384
    c_z: int = 128
    dropout: float = 0.1


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PairBiasAttention(nn.Module):
    def __init__(self, c_s: int, c_z: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = c_s // n_heads
        self.q = nn.Linear(c_s, c_s)
        self.k = nn.Linear(c_s, c_s)
        self.v = nn.Linear(c_s, c_s)
        self.z_bias = nn.Linear(c_z, n_heads)
        self.out = nn.Linear(c_s, c_s)
        self.drop = nn.Dropout(dropout)

    def forward(self, s, z, pair_mask):
        B, L, C = s.shape
        q = self.q(s).view(B, L, self.n_heads, self.head_dim)
        k = self.k(s).view(B, L, self.n_heads, self.head_dim)
        v = self.v(s).view(B, L, self.n_heads, self.head_dim)
        logits = torch.einsum("blhd,bmhd->bhlm", q, k) / math.sqrt(self.head_dim)
        bias = self.z_bias(z).permute(0, 3, 1, 2)
        logits = logits + bias
        if pair_mask is not None:
            mask = pair_mask[:, None, :, :]
            logits = logits.masked_fill(~mask, -1e9)
        attn = torch.softmax(logits, dim=-1)
        attn = self.drop(attn)
        out = torch.einsum("bhlm,bmhd->blhd", attn, v).reshape(B, L, C)
        out = self.out(out)
        return out


class PairRowColAttention(nn.Module):
    def __init__(self, c_z: int, n_heads: int, dropout: float):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(c_z, n_heads, dropout=dropout, batch_first=True)
        self.col_attn = nn.MultiheadAttention(c_z, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(c_z)
        self.drop = nn.Dropout(dropout)

    def forward(self, z, pair_mask):
        B, L, _, C = z.shape
        z_norm = self.norm(z)

        z_row = z_norm.reshape(B * L, L, C)
        row_mask = None
        if pair_mask is not None:
            row_mask = ~pair_mask.reshape(B * L, L)
        row_out, _ = self.row_attn(z_row, z_row, z_row, key_padding_mask=row_mask, need_weights=False)
        row_out = row_out.reshape(B, L, L, C)

        z_col = z_norm.transpose(1, 2).reshape(B * L, L, C)
        col_mask = None
        if pair_mask is not None:
            col_mask = ~pair_mask.transpose(1, 2).reshape(B * L, L)
        col_out, _ = self.col_attn(z_col, z_col, z_col, key_padding_mask=col_mask, need_weights=False)
        col_out = col_out.reshape(B, L, L, C).transpose(1, 2)

        return z + self.drop(row_out + col_out)


class PairformerBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, n_heads: int, dropout: float):
        super().__init__()
        self.s_norm = nn.LayerNorm(c_s)
        self.z_norm = nn.LayerNorm(c_z)
        self.attn = PairBiasAttention(c_s, c_z, n_heads, dropout)
        self.s_ff = FeedForward(c_s, 4, dropout)
        self.pair_attn = PairRowColAttention(c_z, n_heads, dropout)
        self.s_to_z_i = nn.Linear(c_s, c_z, bias=False)
        self.s_to_z_j = nn.Linear(c_s, c_z, bias=False)
        self.z_ff = FeedForward(c_z, 4, dropout)

    def forward(self, s, z, pair_mask):
        s = s + self.attn(self.s_norm(s), self.z_norm(z), pair_mask)
        s = s + self.s_ff(self.s_norm(s))
        z = self.pair_attn(z, pair_mask)
        zi = self.s_to_z_i(self.s_norm(s))
        zj = self.s_to_z_j(self.s_norm(s))
        z = z + zi[:, :, None, :] + zj[:, None, :, :]
        z = z + self.z_ff(self.z_norm(z))
        return s, z


class PairformerStack(nn.Module):
    def __init__(self, n_blocks: int, n_heads: int, c_s: int, c_z: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [PairformerBlock(c_s=c_s, c_z=c_z, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)]
        )

    def forward(self, s, z, pair_mask):
        for blk in self.blocks:
            s, z = blk(s, z, pair_mask)
        return s, z


class PairformerConditioner(nn.Module):
    def __init__(self, c_s_inputs: int, cfg: PairformerCfg = PairformerCfg()):
        super().__init__()
        self.cfg = cfg
        self.s_proj = nn.Linear(c_s_inputs, cfg.c_s)
        self.pairformer = PairformerStack(
            n_blocks=cfg.n_blocks,
            n_heads=cfg.n_heads,
            c_s=cfg.c_s,
            c_z=cfg.c_z,
            dropout=cfg.dropout,
        )

    def forward(self, s_inputs, z_init, pair_mask):
        s = self.s_proj(s_inputs)
        if pair_mask is not None:
            pair_mask = pair_mask.bool()
        s_trunk, z_trunk = self.pairformer(s=s, z=z_init, pair_mask=pair_mask)
        return s_trunk, z_trunk
