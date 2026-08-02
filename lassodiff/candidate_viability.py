"""Coordinate-free sequence/candidate viability head."""
from __future__ import annotations

import torch
import torch.nn as nn


class CandidateViabilityHead(nn.Module):
    def __init__(self, hidden_dim: int = 128, layers: int = 2, heads: int = 4, max_length: int = 256):
        super().__init__()
        self.aa_embedding = nn.Embedding(21, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, heads, 4 * hidden_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.role_embedding = nn.Embedding(5, hidden_dim)
        self.head = nn.Sequential(nn.LayerNorm(3 * hidden_dim), nn.Linear(3 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, aa_ids, token_mask, acceptor_index, plug_index):
        """No coordinate argument is accepted by design."""
        B, L = aa_ids.shape
        if token_mask.shape != (B, L) or acceptor_index.ndim != 2 or plug_index.shape != acceptor_index.shape:
            raise ValueError("viability input shape mismatch")
        valid = token_mask.gather(1, acceptor_index) & token_mask.gather(1, plug_index)
        valid &= plug_index > acceptor_index
        acceptor_aa = aa_ids.gather(1, acceptor_index)
        # seq_encoder IDs: D=2, E=3.
        valid &= (acceptor_aa == 2) | (acceptor_aa == 3)
        if L > self.position_embedding.num_embeddings:
            raise ValueError("sequence exceeds viability position embedding limit")
        position = torch.arange(L, device=aa_ids.device)[None]
        encoded = self.encoder(
            self.aa_embedding(aa_ids) + self.position_embedding(position), src_key_padding_mask=~token_mask,
        )
        pooled = (encoded * token_mask[..., None]).sum(1) / token_mask.sum(1, keepdim=True).clamp_min(1)
        batch = torch.arange(B, device=aa_ids.device)[:, None]
        k_state = encoded[batch, acceptor_index]
        p_state = encoded[batch, plug_index]
        logits = self.head(torch.cat((pooled[:, None].expand_as(k_state), k_state, p_state), dim=-1)).squeeze(-1)
        return logits.masked_fill(~valid, -torch.inf)
