"""Frozen residue-level ESM wrapper for mini_dev."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


class FrozenESMResidueEncoder(nn.Module):
    def __init__(self, model_name: str = "esm2_t30_150M_UR50D", revision: str = "main",
                 output_dim: int | None = None, model=None):
        super().__init__()
        self.model_name = model_name
        self.revision = revision
        self._model = model
        self.output_dim = int(output_dim or getattr(model, "embed_dim", 0) or 0)
        if model is not None:
            self._freeze()

    def _freeze(self):
        if self._model is not None:
            self._model.eval()
            for parameter in self._model.parameters():
                parameter.requires_grad_(False)

    def _load(self):
        if self._model is not None:
            return
        try:
            import esm  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ESM is unavailable; provide cached embeddings or install fair-esm") from exc
        loader = getattr(esm.pretrained, self.model_name, None)
        if loader is None:
            raise RuntimeError(f"unknown ESM model {self.model_name}")
        self._model, alphabet = loader()
        self._alphabet = alphabet
        self.output_dim = int(getattr(self._model, "embed_dim"))
        self._freeze()

    def forward(self, sequences: list[str], token_mask: torch.Tensor) -> torch.Tensor:
        if token_mask.ndim != 2 or token_mask.shape[0] != len(sequences):
            raise ValueError("token_mask must have shape [B,L] matching sequences")
        self._load()
        self._freeze()
        self._model = self._model.to(token_mask.device)
        self._model.eval()
        # A test/dummy encoder may already implement residue-level embeddings.
        if hasattr(self, "_alphabet"):
            batch_converter = self._alphabet.get_batch_converter()
            _, _, tokens = batch_converter([(str(i), sequence) for i, sequence in enumerate(sequences)])
            tokens = tokens.to(token_mask.device)
            layer = int(getattr(self._model, "num_layers", 1))
            with torch.no_grad():
                output = self._model(tokens, repr_layers=[layer], return_contacts=False)["representations"][layer][:, 1:-1]
        else:
            with torch.no_grad():
                output = self._model(sequences, token_mask)
        if output.shape[:2] != token_mask.shape:
            raise ValueError("ESM residue output shape does not match token mask")
        return output.detach() * token_mask[..., None].to(output.dtype)


class CachedESMResidueEncoder(nn.Module):
    """Frozen cache-backed route used by distributed training."""
    def __init__(self, cache_root: str | Path, output_dim: int = 640,
                 encoder_name: str = "esm2_t30_150M_UR50D", encoder_revision: str = "main"):
        super().__init__()
        self.cache_root = Path(cache_root)
        self.output_dim = output_dim
        self.model_name = encoder_name
        self.revision = encoder_revision

    def forward(self, sequences: list[str], token_mask: torch.Tensor) -> torch.Tensor:
        output = token_mask.new_zeros((*token_mask.shape, self.output_dim), dtype=torch.float32)
        for b, sequence in enumerate(sequences):
            key = embedding_cache_key(self.model_name, self.revision, sequence)
            candidates = (self.cache_root / f"{key}.pt", self.cache_root / key / "embedding.pt")
            path = next((x for x in candidates if x.is_file()), None)
            if path is None:
                raise FileNotFoundError(f"missing ESM cache item for sequence {sequence!r}: {key}")
            item = load_embedding_cache_item(path, self.model_name, self.revision)
            embedding = item["embedding"]
            if embedding.ndim != 2 or embedding.shape[1] != self.output_dim or embedding.shape[0] != len(sequence):
                raise RuntimeError("ESM cache embedding shape mismatch")
            output[b, :embedding.shape[0]] = embedding.to(output.device)
        return output * token_mask[..., None]


def embedding_cache_key(encoder_name: str, encoder_revision: str, sequence: str) -> str:
    normalized = "".join(sequence.upper().split())
    payload = f"{encoder_name}\n{encoder_revision}\n{normalized}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_embedding_cache_item(path: str | Path, sequence: str, embedding: torch.Tensor,
                              encoder_name: str, encoder_revision: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "sequence": sequence,
        "embedding": embedding.detach().cpu(),
        "encoder_name": encoder_name,
        "encoder_revision": encoder_revision,
        "sha256": embedding_cache_key(encoder_name, encoder_revision, sequence),
    }, path)


def load_embedding_cache_item(path: str | Path, encoder_name: str, encoder_revision: str) -> dict:
    item = torch.load(path, map_location="cpu", weights_only=False)
    expected = embedding_cache_key(encoder_name, encoder_revision, item["sequence"])
    if item.get("encoder_name") != encoder_name or item.get("encoder_revision") != encoder_revision or item.get("sha256") != expected:
        raise RuntimeError("ESM embedding cache provenance mismatch")
    return item
