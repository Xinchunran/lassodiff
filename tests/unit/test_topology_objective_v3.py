from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.topology_adapter import CandidateBatch
from lassodiff.topology_objective import (
    TopologyLossWeights,
    endpoint_from_velocity,
    topology_supervised_candidate_loss,
)


def _valid_case():
    B, M, L, A = 1, 1, 7, 7
    target = torch.zeros(B, M, L, A, 3)
    ring = torch.tensor([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
    ])
    target[0, 0, :4, 1] = ring
    target[0, 0, 4, 1] = torch.tensor([0.0, 0.0, -2.0])
    target[0, 0, 5, 1] = torch.tensor([0.0, 0.0, -1.0])
    target[0, 0, 6, 1] = torch.tensor([0.0, 0.0, 1.0])
    for residue in range(L):
        ca = target[0, 0, residue, 1]
        target[0, 0, residue, 0] = ca + torch.tensor([0.2, 0.0, 0.0])
        target[0, 0, residue, 2] = ca + torch.tensor([0.0, 0.2, 0.0])
        target[0, 0, residue, 3] = ca + torch.tensor([0.0, 0.0, 0.2])
    # N--Ciso 1.33 A, N-C-O 120 degrees, and the four proxy atoms are planar.
    target[0, 0, 0, 0] = torch.tensor([-1.33, 0.0, 0.0])
    target[0, 0, 1, 1] = torch.tensor([1.0, -1.0, 0.0])
    target[0, 0, 1, 4] = torch.tensor([0.0, 0.0, 0.0])
    target[0, 0, 1, 5] = torch.tensor([0.5, 0.8660254, 0.0])
    atom_mask = torch.ones(B, M, L, A, dtype=torch.bool)
    atom_mask[..., 6] = False
    token_mask = torch.ones(B, L, dtype=torch.bool)
    candidates = CandidateBatch(
        torch.tensor([[3]]), torch.tensor([[4]]), torch.tensor([[1]]),
        torch.tensor([[1.0]]), torch.tensor([[True]]),
    )
    return target, atom_mask, token_mask, candidates


def test_endpoint_reconstruction_matches_rectified_flow_contract():
    target, _atom_mask, _token_mask, _candidates = _valid_case()
    noise = torch.randn_like(target)
    time = torch.tensor([0.35])
    x_t = (1 - time[:, None, None, None, None]) * noise + time[:, None, None, None, None] * target
    endpoint = endpoint_from_velocity(x_t, target - noise, time)
    torch.testing.assert_close(endpoint, target)


def test_topology_objective_accepts_real_one_oxygen_chemistry_and_is_zero_at_target():
    target, atom_mask, token_mask, candidates = _valid_case()
    noise = torch.randn_like(target)
    time = torch.tensor([0.2])
    x_t = (1 - time[:, None, None, None, None]) * noise + time[:, None, None, None, None] * target
    result = topology_supervised_candidate_loss(
        target - noise, target - noise, x_t, time, target,
        token_mask, atom_mask, candidates, TopologyLossWeights(),
    )
    assert result.total.shape == (1, 1)
    assert result.total.item() == pytest.approx(0.0, abs=1e-6)
    assert result.iso_distance.item() == pytest.approx(0.0, abs=1e-6)
    assert result.iso_angle.item() == pytest.approx(0.0, abs=1e-6)
    assert result.iso_plane.item() == pytest.approx(0.0, abs=1e-6)
    assert result.threading.item() == pytest.approx(0.0, abs=1e-6)


def test_broken_iso_and_threading_raise_loss_and_backpropagate():
    target, atom_mask, token_mask, candidates = _valid_case()
    noise = torch.randn_like(target)
    time = torch.tensor([0.25])
    x_t = (1 - time[:, None, None, None, None]) * noise + time[:, None, None, None, None] * target
    velocity = (target - noise).clone()
    # Moving NTERM and the plug/tail changes reactive geometry and link state.
    velocity[0, 0, 0, 0, 0] += 5.0
    velocity[0, 0, 5:, 1, 0] += 4.0
    velocity.requires_grad_()
    result = topology_supervised_candidate_loss(
        velocity, target - noise, x_t, time, target,
        token_mask, atom_mask, candidates, TopologyLossWeights(),
    )
    assert result.iso_distance.item() > 0
    assert result.threading.item() > 0
    result.total.sum().backward()
    assert velocity.grad is not None
    assert torch.isfinite(velocity.grad).all()
    assert velocity.grad.abs().sum() > 0
