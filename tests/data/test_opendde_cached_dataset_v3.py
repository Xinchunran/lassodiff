from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.data.opendde_cached import OpenDDECachedDataset, collate_lassodiff_v3
from lassodiff.opendde_bridge.cache import OpenDDEReasoningCache, ReasoningCacheIdentity
from lassodiff.opendde_bridge.schema import OpenDDEReasoningState


class Base:
    def __init__(self):
        self.items = [self.item("ACDEF", "one"), self.item("ACDEFGH", "two")]

    @staticmethod
    def item(sequence, name):
        L = len(sequence)
        return {
            "record_id": name, "sequence": sequence, "aa_ids": torch.zeros(L, dtype=torch.long),
            "coords": torch.randn(2, L, 7, 3), "atom_mask": torch.ones(2, L, 7, dtype=torch.bool),
            "target_names": ["min1", "min2"], "iso_acceptor_index": 1, "iso_acceptor_type": "ASP",
            "candidates": [
                {"k": 1, "p": 3, "acceptor_index": 1, "prior": .8, "rank": 1},
                {"k": 1, "p": 4, "acceptor_index": 1, "prior": .2, "rank": 2},
            ],
        }

    def __len__(self): return len(self.items)
    def __getitem__(self, index): return self.items[index]


def _identity(sequence):
    return ReasoningCacheIdentity(sequence, "commit", "checkpoint", 1, False, False, "float16", 10)


def _state(identity):
    L = len(identity.sequence)
    return OpenDDEReasoningState(
        torch.randn(1, L, 9), torch.randn(1, L, L, 7), torch.ones(1, L, dtype=torch.bool),
        torch.arange(L)[None], (identity.sequence_sha256,), identity.checkpoint_sha256,
        identity.opendde_commit, identity.feature_schema_version,
    )


def test_cached_dataset_collates_reasoning_and_candidate_targets(tmp_path):
    base, cache = Base(), OpenDDEReasoningCache(tmp_path)
    for item in base.items:
        identity = _identity(item["sequence"])
        cache.put(identity, _state(identity))
    dataset = OpenDDECachedDataset(base, cache, _identity)
    batch = collate_lassodiff_v3([dataset[0], dataset[1]])
    assert batch["reasoning_state"].single.shape == (2, 7, 9)
    assert batch["reasoning_state"].pair.shape == (2, 7, 7, 7)
    assert batch["coords"].shape == (2, 2, 7, 7, 3)
    assert batch["atom_mask"].shape == (2, 2, 7, 7)
    assert batch["target_valid"].all()
    assert batch["acceptor_type"].tolist() == [[0, 0], [0, 0]]
