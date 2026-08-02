from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.candidate_objective import candidate_marginal_loss


def test_prior_and_mask_normalization():
    losses = torch.tensor([[1.0, 2.0, 99.0]])
    prior = torch.tensor([[2.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, False]])
    value, posterior, normalized = candidate_marginal_loss(losses, prior, mask)
    assert torch.isfinite(value)
    torch.testing.assert_close(normalized, torch.tensor([[2 / 3, 1 / 3, 0.0]]))
    assert posterior[0, 2] == 0


def test_candidate_permutation():
    losses = torch.tensor([[1.0, 2.0, 3.0]])
    prior = torch.tensor([[0.2, 0.3, 0.5]])
    mask = torch.tensor([[True, True, True]])
    value, posterior, _ = candidate_marginal_loss(losses, prior, mask)
    perm = torch.tensor([2, 0, 1])
    value_p, posterior_p, _ = candidate_marginal_loss(losses[:, perm], prior[:, perm], mask[:, perm])
    torch.testing.assert_close(value, value_p)
    torch.testing.assert_close(posterior[:, perm], posterior_p)


def test_temperature_limits():
    losses = torch.tensor([[1.0, 3.0]])
    prior = torch.tensor([[0.25, 0.75]])
    mask = torch.tensor([[True, True]])
    low, posterior_low, _ = candidate_marginal_loss(losses, prior, mask, temperature=1e-3)
    high, posterior_high, _ = candidate_marginal_loss(losses, prior, mask, temperature=1e3)
    assert low.item() == pytest.approx(1.0, abs=2e-3)
    assert posterior_low[0, 0] > 0.999
    assert high.item() == pytest.approx((losses * prior).sum().item(), abs=2e-3)
    torch.testing.assert_close(posterior_high, prior, atol=5e-4, rtol=5e-4)
