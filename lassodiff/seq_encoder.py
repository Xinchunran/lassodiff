from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


AA_TO_ID = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
UNK_ID = 20


def seq_to_aa_ids(seq: str) -> torch.Tensor:
    seq = seq.strip().upper()
    ids = [AA_TO_ID.get(a, UNK_ID) for a in seq]
    return torch.tensor(ids, dtype=torch.long)


@dataclass
class SeqEncConfig:
    c_s_inputs: int = 256
    aa_emb_dim: int = 64
    esm_proj_dim: int = 128
    lasso_proj_dim: int = 32
    pos_dim: int = 32


class SequenceEncoder(nn.Module):
    def __init__(self, esm_dim: int, lasso_feat_dim: int, cfg: SeqEncConfig = SeqEncConfig()):
        super().__init__()
        self.cfg = cfg
        self.aa_emb = nn.Embedding(21, cfg.aa_emb_dim)
        self.esm_proj = nn.Linear(esm_dim, cfg.esm_proj_dim)
        self.lasso_proj = nn.Linear(lasso_feat_dim, cfg.lasso_proj_dim)
        self.pos_proj = nn.Linear(cfg.pos_dim, cfg.pos_dim)
        in_dim = cfg.aa_emb_dim + cfg.esm_proj_dim + cfg.lasso_proj_dim + cfg.pos_dim
        self.out = nn.Linear(in_dim, cfg.c_s_inputs)

    def forward(self, aa_ids, esm_residue, lasso_feats, pos1d):
        x = torch.cat(
            [
                self.aa_emb(aa_ids),
                self.esm_proj(esm_residue),
                self.lasso_proj(lasso_feats),
                self.pos_proj(pos1d),
            ],
            dim=-1,
        )
        return self.out(x)
