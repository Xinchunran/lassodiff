from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.geometry_features import DynamicGeometryEncoder
from lassodiff.topology_adapter import CandidateBatch


def _candidate():
    return CandidateBatch(torch.tensor([[2]]), torch.tensor([[4]]), torch.tensor([[1]]),
                          torch.tensor([[1.]]), torch.tensor([[True]]))


def test_local_orientation_and_reactive_features_are_se3_invariant():
    torch.manual_seed(8)
    x = torch.randn(1, 1, 5, 7, 3)
    atoms = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    tokens = torch.ones(1, 5, dtype=torch.bool)
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    encoder = DynamicGeometryEncoder()
    base = encoder(x, atoms, tokens, _candidate())
    transformed = torch.einsum("...c,dc->...d", x, rotation) + torch.tensor([3., -2., 7.])
    changed = encoder(transformed, atoms, tokens, _candidate())
    torch.testing.assert_close(base, changed, atol=2e-5, rtol=2e-5)


def test_reactive_channels_mark_nterm_acceptor_plug_ring_and_tail_ring():
    x = torch.randn(1, 1, 5, 7, 3)
    output = DynamicGeometryEncoder()(x, torch.ones(1, 1, 5, 7, dtype=torch.bool),
                                      torch.ones(1, 5, dtype=torch.bool), _candidate())
    reactive = output[..., -3:]
    assert reactive[0, 0, 0, 1, 0] == 1
    assert reactive[0, 0, 4, 2, 1] == 1
    # p=4 means no residue is in the tail for this length.
    assert reactive[..., 2].sum() == 0


def test_geometry_accepts_bfloat16_coordinates_but_computes_vectors_in_fp32():
    x = torch.randn(1, 1, 5, 7, 3).bfloat16()
    output = DynamicGeometryEncoder()(x, torch.ones(1, 1, 5, 7, dtype=torch.bool),
                                      torch.ones(1, 5, dtype=torch.bool), _candidate())
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
