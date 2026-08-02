from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.topology_adapter import CandidateBatch
from lassodiff.topology_checker import strict_topology_check
from lassodiff.topology_threading import soft_threading_score_ca


def _case(threaded=True):
    L, A = 6, 7
    coords = torch.zeros(1, 1, L, A, 3)
    ring = torch.tensor([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.]])
    coords[0, 0, :4, 1] = ring
    coords[0, 0, 4, 1] = torch.tensor([0., 0., -1. if threaded else 2.])
    coords[0, 0, 5, 1] = torch.tensor([0., 0., 1. if threaded else 3.])
    # Put unused backbone atoms around each CA and build a valid Nterm--ASP closure.
    for residue in range(L):
        ca = coords[0, 0, residue, 1]
        coords[0, 0, residue, 0] = ca + torch.tensor([0.2, 0., 0.])
        coords[0, 0, residue, 2] = ca + torch.tensor([0., 0.2, 0.])
        coords[0, 0, residue, 3] = ca + torch.tensor([0., 0., 0.2])
    coords[0, 0, 0, 0] = torch.tensor([-1.33, 0., 0.])
    coords[0, 0, 1, 4] = torch.tensor([0., 0., 0.])
    coords[0, 0, 1, 5] = torch.tensor([.5, .866, 0.])
    coords[0, 0, 1, 6] = torch.tensor([.5, -.866, 0.])
    mask = torch.ones(1, 1, L, A, dtype=torch.bool)
    candidates = CandidateBatch(torch.tensor([[3]]), torch.tensor([[4]]), torch.tensor([[1]]),
                                torch.tensor([[1.]]), torch.tensor([[True]]))
    return coords, mask, candidates


def test_ring_disk_crossing_changes_when_thread_segment_moves():
    threaded = strict_topology_check(*_case(True), token_mask=torch.ones(1, 6, dtype=torch.bool))
    outside = strict_topology_check(*_case(False), token_mask=torch.ones(1, 6, dtype=torch.bool))
    assert threaded.ring_disk_crossing.item() is True
    assert outside.ring_disk_crossing.item() is False


def test_checker_uses_plug_to_tail_not_preplug_loop():
    coords, mask, _ = _case(True)
    # Make k+1 -> p cross the ring, but keep p -> tail on one side.  The
    # pre-plug loop crossing must not be counted as candidate threading.
    extended = torch.cat((coords, coords[:, :, -1:].clone()), dim=2)
    extended_mask = torch.cat((mask, mask[:, :, -1:].clone()), dim=2)
    extended[0, 0, 4, 1] = torch.tensor([0.0, 0.0, -1.0])
    extended[0, 0, 5, 1] = torch.tensor([0.0, 0.0, 1.0])
    extended[0, 0, 6, 1] = torch.tensor([0.0, 0.0, 2.0])
    candidates = CandidateBatch(
        torch.tensor([[3]]), torch.tensor([[5]]), torch.tensor([[1]]),
        torch.tensor([[1.0]]), torch.tensor([[True]]),
    )
    result = strict_topology_check(
        extended, extended_mask, candidates, token_mask=torch.ones(1, 7, dtype=torch.bool),
    )
    assert result.crossing_count.item() == 0
    assert result.threading_success.item() is False


def test_double_crossing_is_counted_and_rejected():
    coords, mask, _ = _case(True)
    coords = torch.cat((coords, coords[:, :, -1:].clone()), dim=2)
    mask = torch.cat((mask, mask[:, :, -1:].clone()), dim=2)
    coords[0, 0, 4, 1] = torch.tensor([0.0, 0.0, -1.0])
    coords[0, 0, 5, 1] = torch.tensor([0.0, 0.0, 1.0])
    coords[0, 0, 6, 1] = torch.tensor([0.0, 0.0, -1.0])
    candidates = CandidateBatch(
        torch.tensor([[3]]), torch.tensor([[4]]), torch.tensor([[1]]),
        torch.tensor([[1.0]]), torch.tensor([[True]]),
    )
    result = strict_topology_check(coords, mask, candidates, token_mask=torch.ones(1, 7, dtype=torch.bool))
    assert result.crossing_count.item() == 2
    assert result.threading_success.item() is False


def test_checker_is_rigid_transform_invariant_and_surrogate_tracks_crossing():
    coords, mask, candidates = _case(True)
    outside, _, _ = _case(False)
    token_mask = torch.ones(1, 6, dtype=torch.bool)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = coords @ rotation.T + torch.tensor([7.0, -3.0, 2.0])
    original = strict_topology_check(coords, mask, candidates, token_mask=token_mask)
    moved = strict_topology_check(transformed, mask, candidates, token_mask=token_mask)
    torch.testing.assert_close(original.crossing_count, moved.crossing_count)
    torch.testing.assert_close(original.signed_crossing, moved.signed_crossing)
    threaded_score = soft_threading_score_ca(coords[..., 1, :], token_mask, candidates)
    outside_score = soft_threading_score_ca(outside[..., 1, :], token_mask, candidates)
    assert threaded_score.abs().item() > outside_score.abs().item() + 0.5


def test_iso_components_are_reported_separately():
    coords, mask, candidates = _case(True)
    # Real relaxed isopeptide PDBs contain one carbonyl oxygen: the second
    # carboxylate oxygen has been replaced by the N-terminal amide nitrogen.
    mask[0, 0, candidates.acceptor_index.item(), 6] = False
    result = strict_topology_check(coords, mask, candidates, token_mask=torch.ones(1, 6, dtype=torch.bool))
    assert result.iso_distance.shape == (1, 1)
    assert result.iso_distance.item() == pytest.approx(1.33, abs=1e-4)
    assert result.iso_distance_valid.item() is True
    assert result.iso_angle_valid.item() is True
    assert result.iso_plane_valid.item() is True


def test_missing_candidate_thread_ca_is_invalid_not_negative():
    coords, mask, candidates = _case(True)
    mask[0, 0, 5, 1] = False
    result = strict_topology_check(coords, mask, candidates, token_mask=torch.ones(1, 6, dtype=torch.bool))
    assert result.checker_valid.item() is False
    assert result.threading_success.item() is False
