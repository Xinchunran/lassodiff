from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn


@dataclass
class ScoreCfg:
    sigma_data: float = 16.0
    c_s: int = 384
    c_z: int = 128
    c_s_inputs: int = 256
    c_a: int = 128
    n_blocks: int = 6
    n_heads: int = 8
    use_protenix: bool = False
    n_atom: int = 7


def make_relp_feature_single_chain(
    L: int, device: torch.device, r_max: int = 32, s_max: int = 2
) -> torch.Tensor:
    idx = torch.arange(L, device=device)
    asym_id = torch.zeros(L, device=device, dtype=torch.long)
    residue_index = idx.clone()
    entity_id = torch.zeros(L, device=device, dtype=torch.long)
    token_index = idx.clone()
    sym_id = torch.zeros(L, device=device, dtype=torch.long)

    b_same_chain = (asym_id[:, None] == asym_id[None, :]).long()
    b_same_res = (residue_index[:, None] == residue_index[None, :]).long()
    b_same_ent = (entity_id[:, None] == entity_id[None, :]).long()

    d_res = (
        torch.clip(residue_index[:, None] - residue_index[None, :] + r_max, 0, 2 * r_max)
        * b_same_chain
        + (1 - b_same_chain) * (2 * r_max + 1)
    )
    a_rel_pos = torch.nn.functional.one_hot(d_res, 2 * (r_max + 1))

    d_tok = (
        torch.clip(token_index[:, None] - token_index[None, :] + r_max, 0, 2 * r_max)
        * b_same_chain
        * b_same_res
        + (1 - b_same_chain * b_same_res) * (2 * r_max + 1)
    )
    a_rel_tok = torch.nn.functional.one_hot(d_tok, 2 * (r_max + 1))

    d_chain = (
        torch.clip(sym_id[:, None] - sym_id[None, :] + s_max, 0, 2 * s_max)
        * b_same_ent
        + (1 - b_same_ent) * (2 * s_max + 1)
    )
    a_rel_chain = torch.nn.functional.one_hot(d_chain, 2 * (s_max + 1))

    relp = torch.cat([a_rel_pos, a_rel_tok, b_same_ent[..., None], a_rel_chain], dim=-1).float()
    return relp.unsqueeze(0)


class PairBiasAttention(nn.Module):
    def __init__(self, c_a: int, c_z: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = c_a // n_heads
        self.q = nn.Linear(c_a, c_a)
        self.k = nn.Linear(c_a, c_a)
        self.v = nn.Linear(c_a, c_a)
        self.z_bias = nn.Linear(c_z, n_heads)
        self.out = nn.Linear(c_a, c_a)
        self.drop = nn.Dropout(dropout)

    def forward(self, a, z, pair_mask):
        B, L, C = a.shape
        q = self.q(a).view(B, L, self.n_heads, self.head_dim)
        k = self.k(a).view(B, L, self.n_heads, self.head_dim)
        v = self.v(a).view(B, L, self.n_heads, self.head_dim)
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


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, c_a: int, c_z: int, n_heads: int, dropout: float):
        super().__init__()
        self.a_norm = nn.LayerNorm(c_a)
        self.z_norm = nn.LayerNorm(c_z)
        self.attn = PairBiasAttention(c_a, c_z, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(c_a, c_a * 4),
            nn.GELU(),
            nn.Linear(c_a * 4, c_a),
            nn.Dropout(dropout),
        )

    def forward(self, a, z, pair_mask):
        a = a + self.attn(self.a_norm(a), self.z_norm(z), pair_mask)
        a = a + self.ff(self.a_norm(a))
        return a


class DiffusionTransformerSmall(nn.Module):
    def __init__(self, c_a: int, c_z: int, n_blocks: int, n_heads: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DiffusionTransformerBlock(c_a=c_a, c_z=c_z, n_heads=n_heads, dropout=dropout) for _ in range(n_blocks)]
        )

    def forward(self, a, z, pair_mask):
        for blk in self.blocks:
            a = blk(a, z, pair_mask)
        return a


class ResidueScoreHead(nn.Module):
    def __init__(self, cfg: ScoreCfg = ScoreCfg()):
        super().__init__()
        self.cfg = cfg
        self.s_in_proj = nn.Linear(cfg.c_s_inputs, cfg.c_s)
        self.relp_proj = nn.Linear(2 * (32 + 1) + 2 * (32 + 1) + 1 + 2 * (2 + 1), cfg.c_z)
        self.t_mlp = nn.Sequential(
            nn.Linear(1, cfg.c_s),
            nn.SiLU(),
            nn.Linear(cfg.c_s, cfg.c_s),
        )
        self.x_proj = nn.Linear(3 * cfg.n_atom, cfg.c_a)
        self.s_to_a = nn.Linear(cfg.c_s, cfg.c_a)
        self.use_protenix = cfg.use_protenix
        if self.use_protenix:
            from .protenix_modules import DiffusionConditioning, DiffusionTransformer

            self.diffusion_conditioning = DiffusionConditioning(
                sigma_data=cfg.sigma_data,
                c_z=cfg.c_z,
                c_s=cfg.c_s,
                c_s_inputs=cfg.c_s_inputs,
            )
            self.tr = DiffusionTransformer(
                c_a=cfg.c_a,
                c_s=cfg.c_s,
                c_z=cfg.c_z,
                n_blocks=cfg.n_blocks,
                n_heads=cfg.n_heads,
            )
        else:
            self.tr = DiffusionTransformerSmall(
                c_a=cfg.c_a,
                c_z=cfg.c_z,
                n_blocks=cfg.n_blocks,
                n_heads=cfg.n_heads,
                dropout=0.1,
            )
        self.out = nn.Linear(cfg.c_a, 3 * cfg.n_atom)

    def forward(self, x_t, sigma, s_inputs, s_trunk, z_trunk, token_mask):
        B, Ns, L, A, _ = x_t.shape
        device = x_t.device

        relp_feature = make_relp_feature_single_chain(L, device=device)
        relp_feature = relp_feature.repeat(B, 1, 1, 1)

        if self.use_protenix:
            single_s, pair_z = self.diffusion_conditioning(
                t_hat_noise_level=sigma,
                relp_feature=relp_feature,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=None,
                inplace_safe=False,
                use_conditioning=True,
            )
            x_flat = x_t.reshape(B, Ns, L, A * 3)
            a = self.x_proj(x_flat) + self.s_to_a(single_s)
            a2 = a.reshape(B * Ns, L, -1)
            s2 = single_s.reshape(B * Ns, L, -1)
            z2 = pair_z.unsqueeze(1).repeat(1, Ns, 1, 1, 1).reshape(B * Ns, L, L, -1)
            a_out = self.tr(a=a2, s=s2, z=z2)
            v = self.out(a_out).reshape(B, Ns, L, A, 3)
        else:
            relp = self.relp_proj(relp_feature)
            pair_z = z_trunk + relp

            s_cond = s_trunk + self.s_in_proj(s_inputs)
            t_feat = self.t_mlp(sigma.reshape(B * Ns, 1)).reshape(B, Ns, -1)
            single_s = s_cond[:, None, :, :] + t_feat[:, :, None, :]

            x_flat = x_t.reshape(B, Ns, L, A * 3)
            a = self.x_proj(x_flat) + self.s_to_a(single_s)

            a2 = a.reshape(B * Ns, L, -1)
            z2 = pair_z.unsqueeze(1).repeat(1, Ns, 1, 1, 1).reshape(B * Ns, L, L, -1)
            mask = token_mask[:, None, :].repeat(1, Ns, 1).reshape(B * Ns, L)
            pair_mask = mask[:, :, None] & mask[:, None, :]

            a_out = self.tr(a2, z2, pair_mask)
            v = self.out(a_out).reshape(B, Ns, L, A, 3)
        v = v * token_mask[:, None, :, None, None].float()
        return v


ProtenixResidueScoreHead = ResidueScoreHead
