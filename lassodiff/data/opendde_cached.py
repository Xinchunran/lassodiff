"""Candidate-specific LassoPred targets joined to sequence-keyed OpenDDE states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.utils.data import Dataset

from .lassopred_lmdb import collate_lassopred_v2
from ..opendde_bridge.cache import OpenDDEReasoningCache, ReasoningCacheIdentity
from ..opendde_bridge.schema import OpenDDEReasoningState, collate_reasoning_states


@dataclass(frozen=True)
class CandidateStructureTargets:
    coords: torch.Tensor
    atom_mask: torch.Tensor
    target_valid: torch.Tensor


class OpenDDECachedDataset(Dataset):
    def __init__(
        self, base: Dataset, cache: OpenDDEReasoningCache,
        identity_builder: Callable[[str], ReasoningCacheIdentity],
    ):
        self.base = base
        self.cache = cache
        self.identity_builder = identity_builder

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        identity = self.identity_builder(item["sequence"])
        item["reasoning_state"] = self.cache.get(identity)
        return item

    def set_epoch(self, epoch):
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)


def collate_lassodiff_v3(items):
    if not items:
        raise ValueError("cannot collate an empty V3 batch")
    batch = collate_lassopred_v2(items)
    reasoning = collate_reasoning_states([item["reasoning_state"] for item in items])
    if reasoning.token_mask.shape != batch["token_mask"].shape or not torch.equal(
        reasoning.token_mask, batch["token_mask"]
    ):
        raise RuntimeError("OpenDDE reasoning token mask does not match LassoPred sequence padding")
    B, M = batch["candidate_mask"].shape
    acceptor_type = torch.zeros((B, M), dtype=torch.long)
    for b, item in enumerate(items):
        value = {"ASP": 0, "GLU": 1}.get(str(item["iso_acceptor_type"]).upper())
        if value is None:
            raise RuntimeError("V3 candidate acceptor type must be ASP or GLU")
        acceptor_type[b, :len(item["candidates"])] = value
    batch["acceptor_type"] = acceptor_type
    if "target_valid" not in batch:
        raise RuntimeError("V3 structure batch requires explicit target_valid provenance")
    if not bool((batch["target_valid"] <= batch["candidate_mask"]).all()):
        raise RuntimeError("target_valid cannot enable a padded or illegal candidate")
    batch["reasoning_state"] = reasoning
    return batch
