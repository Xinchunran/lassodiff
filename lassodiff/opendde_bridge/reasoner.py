from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from typing import Any
from contextlib import nullcontext

import torch
import torch.nn as nn

from .schema import OpenDDEReasoningState, validate_reasoning_state


class FrozenOpenDDEReasoner(nn.Module):
    """Own a frozen model and expose only a validated reasoning-state API."""

    def __init__(
        self,
        model: nn.Module,
        forward_reasoning: Callable[[Mapping[str, torch.Tensor]], OpenDDEReasoningState],
    ) -> None:
        super().__init__()
        self.model = model
        self._forward_reasoning = forward_reasoning
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, features: Mapping[str, torch.Tensor]) -> OpenDDEReasoningState:
        self.model.eval()
        return validate_reasoning_state(self._forward_reasoning(features))


class CacheOnlyOpenDDEReasoner(nn.Module):
    """Training placeholder that can never silently compute/fallback.

    Formal training consumes provenance-checked offline reasoning states.  The
    real 656M frozen trunk is verified by preflight and cache generation, but
    is intentionally not replicated in every FSDP rank.
    """

    def __init__(self, checkpoint_sha256: str, opendde_commit: str):
        super().__init__()
        self.checkpoint_sha256 = checkpoint_sha256
        self.opendde_commit = opendde_commit

    def forward(self, _features):
        raise RuntimeError("cache-only OpenDDE reasoner cannot run forward; populate the strict reasoning cache")


class PinnedOpenDDEForward:
    """Versioned adapter around official ``get_pairformer_output``.

    This class intentionally receives an already strict-loaded official model;
    construction/loading is kept outside so tests can prove both contracts
    independently and no import fallback is possible.
    """

    def __init__(
        self, model, checkpoint_sha256: str, opendde_commit: str, n_cycle: int = 10,
        inference_dtype: str = "fp32",
    ):
        if not hasattr(model, "get_pairformer_output"):
            raise RuntimeError("Pinned OpenDDE model has no get_pairformer_output API")
        self.model = model
        self.checkpoint_sha256 = checkpoint_sha256
        self.opendde_commit = opendde_commit
        self.n_cycle = int(n_cycle)
        self.inference_dtype = str(inference_dtype)

    @staticmethod
    def sequence_hash(sequence: str) -> str:
        return hashlib.sha256(sequence.encode("ascii")).hexdigest()

    @torch.no_grad()
    def __call__(self, features: Mapping[str, Any]) -> OpenDDEReasoningState:
        from opendde.model.opendde import update_input_feature_dict

        sequence = str(features.get("_sequence", ""))
        if not sequence:
            raise ValueError("OpenDDE feature dictionary is missing _sequence provenance")
        device = next(self.model.parameters()).device
        model_features = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in features.items() if not key.startswith("_")
        }
        model_features = self.model.relative_position_encoding.generate_relp(model_features, lazy=False)
        model_features = update_input_feature_dict(model_features)
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda" and self.inference_dtype == "bf16" else nullcontext()
        )
        with amp:
            _s_inputs, single, pair = self.model.get_pairformer_output(
                model_features, N_cycle=self.n_cycle, inplace_safe=False, chunk_size=None
            )
        if single.ndim == 2:
            single, pair = single[None], pair[None]
        B, L = single.shape[:2]
        token_mask = torch.ones((B, L), dtype=torch.bool, device=single.device)
        residue_index = torch.arange(L, device=single.device)[None].expand(B, -1)
        from lassodiff.seq_encoder import seq_to_aa_ids
        residue_type = seq_to_aa_ids(sequence).to(single.device)[None].expand(B, -1)
        return OpenDDEReasoningState(
            single=single.float(), pair=pair.float(), token_mask=token_mask, residue_index=residue_index,
            sequence_hashes=(self.sequence_hash(sequence),) * B,
            checkpoint_sha256=self.checkpoint_sha256, opendde_commit=self.opendde_commit,
            feature_schema_version=1,
            residue_type=residue_type,
        )
