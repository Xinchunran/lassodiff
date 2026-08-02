from __future__ import annotations

from dataclasses import dataclass

import torch


FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OpenDDEReasoningState:
    single: torch.Tensor
    pair: torch.Tensor
    token_mask: torch.Tensor
    residue_index: torch.Tensor
    sequence_hashes: tuple[str, ...]
    checkpoint_sha256: str
    opendde_commit: str
    feature_schema_version: int
    residue_type: torch.Tensor | None = None

    def to(self, device: torch.device | str, *, dtype: torch.dtype | None = None):
        return OpenDDEReasoningState(
            single=self.single.to(device=device, dtype=dtype or self.single.dtype),
            pair=self.pair.to(device=device, dtype=dtype or self.pair.dtype),
            token_mask=self.token_mask.to(device=device), residue_index=self.residue_index.to(device=device),
            sequence_hashes=self.sequence_hashes, checkpoint_sha256=self.checkpoint_sha256,
            opendde_commit=self.opendde_commit, feature_schema_version=self.feature_schema_version,
            residue_type=None if self.residue_type is None else self.residue_type.to(device=device),
        )


def validate_reasoning_state(state: OpenDDEReasoningState) -> OpenDDEReasoningState:
    if state.single.ndim != 3:
        raise ValueError("OpenDDE single must be [B,L,C_s]")
    B, L = state.single.shape[:2]
    if state.pair.ndim != 4 or tuple(state.pair.shape[:3]) != (B, L, L):
        raise ValueError("OpenDDE pair must be [B,L,L,C_z]")
    if tuple(state.token_mask.shape) != (B, L) or state.token_mask.dtype != torch.bool:
        raise ValueError("OpenDDE token_mask must be bool [B,L]")
    if tuple(state.residue_index.shape) != (B, L):
        raise ValueError("OpenDDE residue_index must be [B,L]")
    if state.residue_type is not None and tuple(state.residue_type.shape) != (B, L):
        raise ValueError("OpenDDE residue_type must be [B,L]")
    if len(state.sequence_hashes) != B or any(not value for value in state.sequence_hashes):
        raise ValueError("OpenDDE sequence hashes are invalid")
    if not state.checkpoint_sha256 or not state.opendde_commit:
        raise ValueError("OpenDDE reasoning provenance is required")
    if state.feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"OpenDDE feature schema {state.feature_schema_version} != {FEATURE_SCHEMA_VERSION}"
        )
    if not torch.isfinite(state.single).all() or not torch.isfinite(state.pair).all():
        raise ValueError("OpenDDE reasoning contains non-finite values")
    invalid_single = ~state.token_mask[..., None]
    pair_mask = state.token_mask[:, :, None] & state.token_mask[:, None, :]
    if bool(state.single.masked_select(invalid_single.expand_as(state.single)).ne(0).any()):
        raise ValueError("OpenDDE padding single must be zero")
    if bool(state.pair.masked_select(~pair_mask[..., None].expand_as(state.pair)).ne(0).any()):
        raise ValueError("OpenDDE padding pair must be zero")
    return state


def collate_reasoning_states(states: list[OpenDDEReasoningState]) -> OpenDDEReasoningState:
    if not states:
        raise ValueError("cannot collate an empty OpenDDE reasoning-state list")
    states = [validate_reasoning_state(state) for state in states]
    provenance = {
        (state.checkpoint_sha256, state.opendde_commit, state.feature_schema_version)
        for state in states
    }
    if len(provenance) != 1:
        raise ValueError("cannot collate OpenDDE states with different provenance")
    if any(state.single.shape[0] != 1 for state in states):
        raise ValueError("cached OpenDDE states must contain exactly one sequence")
    c_s = states[0].single.shape[-1]
    c_z = states[0].pair.shape[-1]
    if any(state.single.shape[-1] != c_s or state.pair.shape[-1] != c_z for state in states):
        raise ValueError("OpenDDE reasoning channel dimensions differ")
    B, L = len(states), max(state.single.shape[1] for state in states)
    single = states[0].single.new_zeros((B, L, c_s))
    pair = states[0].pair.new_zeros((B, L, L, c_z))
    token_mask = torch.zeros((B, L), dtype=torch.bool, device=single.device)
    residue_index = torch.zeros((B, L), dtype=torch.long, device=single.device)
    residue_type = torch.full((B, L), 20, dtype=torch.long, device=single.device)
    has_residue_type = all(state.residue_type is not None for state in states)
    hashes = []
    for index, state in enumerate(states):
        length = state.single.shape[1]
        single[index, :length] = state.single[0]
        pair[index, :length, :length] = state.pair[0]
        token_mask[index, :length] = state.token_mask[0]
        residue_index[index, :length] = state.residue_index[0]
        if has_residue_type:
            residue_type[index, :length] = state.residue_type[0]
        hashes.append(state.sequence_hashes[0])
    checkpoint, commit, schema = provenance.pop()
    return validate_reasoning_state(OpenDDEReasoningState(
        single=single, pair=pair, token_mask=token_mask, residue_index=residue_index,
        sequence_hashes=tuple(hashes), checkpoint_sha256=checkpoint,
        opendde_commit=commit, feature_schema_version=schema,
        residue_type=residue_type if has_residue_type else None,
    ))
