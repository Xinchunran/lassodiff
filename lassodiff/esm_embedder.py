from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ESMConfig:
    model_name: str = "esm2_t12_35M_UR50D"
    repr_layer: Optional[int] = None
    use_fp16: bool = True
    max_len: int = 1022
    device: Optional[str] = None
    freeze: bool = True


class ESMEmbedder(nn.Module):
    def __init__(self, cfg: ESMConfig):
        super().__init__()
        self.cfg = cfg
        self.device_str = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        import esm

        if hasattr(esm.pretrained, cfg.model_name):
            loader = getattr(esm.pretrained, cfg.model_name)
            model, alphabet = loader()
        else:
            model, alphabet = esm.pretrained.load_model_and_alphabet(cfg.model_name)

        self.model = model.to(self.device_str)
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.model.eval()

        if cfg.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        self.pad_idx = self.alphabet.padding_idx

        with torch.no_grad():
            tmp = self.forward(["ACDE"], return_pooled=False)
        self.esm_dim = tmp["esm_residue"].shape[-1]

    @torch.no_grad()
    def forward(self, seqs: List[str], return_pooled: bool = True) -> Dict[str, torch.Tensor]:
        seqs = [s.strip().upper().replace(" ", "") for s in seqs]
        for s in seqs:
            if len(s) > self.cfg.max_len:
                raise ValueError(f"Sequence too long: {len(s)} > {self.cfg.max_len}")

        data = [(f"seq{i}", s) for i, s in enumerate(seqs)]
        _, _, toks = self.batch_converter(data)
        toks = toks.to(self.device_str)

        repr_layer = self.model.num_layers if self.cfg.repr_layer is None else self.cfg.repr_layer

        use_amp = self.cfg.use_fp16 and self.device_str.startswith("cuda")
        autocast = torch.autocast("cuda", dtype=torch.float16) if use_amp else _NullCtx()

        with autocast:
            out = self.model(toks, repr_layers=[repr_layer], return_contacts=False)
            reps = out["representations"][repr_layer]

        tok_mask = toks.ne(self.pad_idx)
        reps = reps[:, 1:-1, :]
        tok_mask = tok_mask[:, 1:-1]

        esm_residue = reps
        esm_mask = tok_mask.bool()

        out_dict = {"esm_residue": esm_residue, "esm_mask": esm_mask}
        if return_pooled:
            denom = esm_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
            pooled = (esm_residue * esm_mask.unsqueeze(-1)).sum(dim=1) / denom
            out_dict["esm_pooled"] = pooled
        return out_dict


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
