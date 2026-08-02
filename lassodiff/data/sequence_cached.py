"""Strict OpenDDE-cache dataset for verified sequence gate examples."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .sequence_v3 import SequenceExample
from ..opendde_bridge.schema import collate_reasoning_states


class CachedSequenceDataset(Dataset):
    def __init__(self, examples, cache, identity_builder):
        self.examples = list(examples)
        self.cache, self.identity_builder = cache, identity_builder

    def __len__(self): return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return example, self.cache.get(self.identity_builder(example.sequence))


def collate_sequence_gate(items, max_candidates=3):
    examples, states = zip(*items)
    state = collate_reasoning_states(list(states))
    B = len(examples)
    label = torch.tensor([example.label for example in examples], dtype=torch.float32)
    acceptor = torch.tensor([example.acceptor_index if example.acceptor_index is not None else -1 for example in examples])
    plug = torch.tensor([example.plug_index if example.plug_index is not None else -1 for example in examples])
    prior = torch.zeros(B, max_candidates)
    candidate_mask = torch.zeros(B, max_candidates, dtype=torch.bool)
    for index, example in enumerate(examples):
        if example.teacher_prior is None:
            continue
        if len(example.teacher_prior) > max_candidates:
            raise RuntimeError("sequence teacher prior exceeds max_candidates")
        values = torch.tensor(example.teacher_prior, dtype=torch.float32)
        values = values / values.sum()
        prior[index, :len(values)] = values
        candidate_mask[index, :len(values)] = True
    return {
        "example_ids": [example.example_id for example in examples], "reasoning_state": state,
        "label": label, "acceptor_index": acceptor, "plug_index": plug,
        "teacher_prior": prior, "candidate_mask": candidate_mask,
        "ood_target": torch.tensor([example.ood for example in examples], dtype=torch.float32),
        "kind": [example.kind for example in examples],
    }
