from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.score_head_lasso_v3 import EquivariantLassoDiffusionV3


def test_velocity_rotation_equivariance_and_translation_invariance():
    torch.manual_seed(12)
    head = EquivariantLassoDiffusionV3(16, 12, 24, n_blocks=2, n_heads=4).eval()
    B, M, L, A = 1, 2, 5, 7
    x = torch.randn(B, M, L, A, 3)
    s = torch.randn(B, M, L, 16)
    z = torch.randn(B, M, L, L, 12)
    tokens = torch.ones(B, L, dtype=torch.bool)
    atoms = torch.ones(B, M, L, A, dtype=torch.bool)
    pairs = torch.ones(B, M, L, L, dtype=torch.bool)
    t = torch.rand(B)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    base = head(x, t, s, z, tokens, atoms, pairs)
    rotated_x = torch.einsum("...c,dc->...d", x, rotation)
    rotated = head(rotated_x, t, s, z, tokens, atoms, pairs)
    expected = torch.einsum("...c,dc->...d", base, rotation)
    torch.testing.assert_close(rotated, expected, atol=3e-5, rtol=3e-5)
    translated = head(x + torch.tensor([7.0, -4.0, 2.0]), t, s, z, tokens, atoms, pairs)
    torch.testing.assert_close(base, translated, atol=3e-5, rtol=3e-5)


def test_geometry_is_recomputed_per_block(monkeypatch):
    head = EquivariantLassoDiffusionV3(16, 12, 24, n_blocks=3, n_heads=4)
    calls = 0
    original = head.geometry_encoder.forward

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(head.geometry_encoder, "forward", counted)
    head(
        torch.randn(1, 1, 4, 7, 3), torch.rand(1), torch.randn(1, 1, 4, 16),
        torch.randn(1, 1, 4, 4, 12), torch.ones(1, 4, dtype=torch.bool),
        torch.ones(1, 1, 4, 7, dtype=torch.bool), torch.ones(1, 1, 4, 4, dtype=torch.bool),
    )
    assert calls == 3
