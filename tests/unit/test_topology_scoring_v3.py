from __future__ import annotations

import pytest
import torch

from lassodiff.topology_adapter import CandidateBatch
from lassodiff.topology_checker import strict_topology_check
from lassodiff.topology_scoring import score_candidates


def _coords():
    B, M, L, A = 1, 2, 7, 7
    coords = torch.zeros(B, M, L, A, 3)
    ring = torch.tensor([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
    coords[:, :, :4, 1] = ring
    coords[0, 0, 4:, 1] = torch.tensor([[0., 0., -1.], [0., 0., 1.], [0., 0., 2.]])
    coords[0, 1, 4:, 1] = torch.tensor([[3., 0., -1.], [3., 0., 1.], [3., 0., 2.]])
    for m in range(M):
        for residue in range(L):
            ca = coords[0, m, residue, 1]
            coords[0, m, residue, 0] = ca + torch.tensor([.2, 0., 0.])
            coords[0, m, residue, 2] = ca + torch.tensor([0., .2, 0.])
            coords[0, m, residue, 3] = ca + torch.tensor([0., 0., .2])
    coords[:, :, 0, 0] = torch.tensor([-1.33, 0., 0.])
    coords[:, :, 1, 4] = torch.tensor([0., 0., 0.])
    coords[:, :, 1, 5] = torch.tensor([.5, .866, 0.])
    mask = torch.ones(B, M, L, A, dtype=torch.bool)
    mask[..., 6] = False
    candidates = CandidateBatch(
        torch.tensor([[3, 3]]), torch.tensor([[4, 4]]), torch.tensor([[1, 1]]),
        torch.tensor([[.5, .5]]), torch.ones(B, M, dtype=torch.bool),
    )
    return coords, mask, candidates


def test_scoring_hard_rejects_unthreaded_candidate_with_reason():
    coords, mask, candidates = _coords()
    result = strict_topology_check(coords, mask, candidates, token_mask=torch.ones(1, 7, dtype=torch.bool))
    scored = score_candidates(
        result, structure_quality=torch.ones(1, 2), tail_clearance_min=0.0,
        severe_clash_rate=.3,
    )
    assert scored.hard_valid.tolist() == [[True, False]]
    assert scored.score[0, 0].item() > -float("inf")
    assert "threading" in scored.rejection_reason[0][1]


def test_scoring_preserves_candidate_dimension_and_explicit_score_components():
    coords, mask, candidates = _coords()
    result = strict_topology_check(coords, mask, candidates, token_mask=torch.ones(1, 7, dtype=torch.bool))
    scored = score_candidates(
        result, structure_quality=torch.tensor([[.9, .1]]), tail_clearance_min=0.0,
        severe_clash_rate=.3,
    )
    assert scored.hard_valid.shape == (1, 2)
    assert scored.score.shape == (1, 2)
    assert scored.rejection_reason[0][0] == ()
