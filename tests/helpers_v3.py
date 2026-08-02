from __future__ import annotations

import torch

from lassodiff.topology_adapter import CandidateBatch


def tiny_candidate_batch(*, candidates_count: int = 2, length: int = 6):
    if not 1 <= candidates_count <= 3:
        raise ValueError("V3 fixture supports one to three candidates")
    if length < candidates_count + 3:
        raise ValueError("sequence is too short for distinct candidate plugs")
    p = torch.arange(3, 3 + candidates_count).unsqueeze(0)
    prior = torch.full((1, candidates_count), 1.0 / candidates_count)
    candidates = CandidateBatch(
        k=torch.ones((1, candidates_count), dtype=torch.long),
        p=p,
        acceptor_index=torch.full((1, candidates_count), 2, dtype=torch.long),
        prior=prior,
        candidate_mask=torch.ones((1, candidates_count), dtype=torch.bool),
        acceptor_type=torch.zeros((1, candidates_count), dtype=torch.long),
    )
    return {
        "token_mask": torch.ones((1, length), dtype=torch.bool),
        "candidates": candidates,
    }
