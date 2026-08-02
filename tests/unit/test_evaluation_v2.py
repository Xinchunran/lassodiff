from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.evaluation_v2 import evaluate_generated_structures, masked_aligned_rmsd
from lassodiff.topology_adapter import CandidateBatch


def _closed_lasso_batch():
    # A small non-planar ring and a tail passing through it.  The exact shape
    # is less important than deterministic target-relative metric behaviour.
    B, M, L, A = 1, 2, 6, 7
    coords = torch.zeros(B, M, L, A, 3)
    ca = torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0],
    ])
    coords[:, :, :, 1] = ca
    coords[:, :, :, 0] = ca + torch.tensor([0.1, 0.0, 0.0])
    coords[:, :, :, 2] = ca + torch.tensor([0.0, 0.1, 0.0])
    coords[:, :, :, 3] = ca + torch.tensor([0.0, 0.0, 0.1])
    coords[:, :, 2, 4] = coords[:, :, 0, 0] + torch.tensor([1.35, 0.0, 0.0])
    mask = torch.ones(B, M, L, A, dtype=torch.bool)
    candidates = CandidateBatch(
        k=torch.tensor([[2, 2]]), p=torch.tensor([[3, 4]]),
        acceptor_index=torch.tensor([[2, 2]]), prior=torch.tensor([[0.5, 0.5]]),
        candidate_mask=torch.tensor([[True, True]]),
    )
    return coords, mask, candidates


def test_masked_aligned_rmsd_removes_rigid_transform():
    target, mask, _ = _closed_lasso_batch()
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred = torch.einsum("...c,dc->...d", target, rotation) + torch.tensor([8.0, -2.0, 3.0])
    rmsd = masked_aligned_rmsd(pred, target, mask)
    torch.testing.assert_close(rmsd, torch.zeros_like(rmsd), atol=2e-4, rtol=0.0)


def test_perfect_generation_matches_native_topology():
    target, mask, candidates = _closed_lasso_batch()
    metrics = evaluate_generated_structures(
        target, target, torch.ones(1, 6, dtype=torch.bool), mask, candidates
    )
    assert metrics["candidate_count"] == 2
    assert metrics["ca_rmsd"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["backbone_rmsd"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["iso_distance_mae"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["link_class_accuracy"] == pytest.approx(1.0)
    assert metrics["topology_match_rate"] == pytest.approx(1.0)
