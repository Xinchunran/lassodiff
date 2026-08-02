from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from lassodiff.opendde_bridge.schema import OpenDDEReasoningState
from lassodiff.sampler_v3 import sample_rectified_flow_v3
from lassodiff.topology_adapter import CandidateBatch


def _state():
    return OpenDDEReasoningState(
        torch.zeros(1, 5, 4), torch.zeros(1, 5, 5, 3), torch.ones(1, 5, dtype=torch.bool),
        torch.arange(5)[None], ("s",), "h", "c", 1,
    )


def _candidates():
    return CandidateBatch(torch.tensor([[1, 1]]), torch.tensor([[3, 4]]), torch.tensor([[2, 2]]),
                          torch.tensor([[.5, .5]]), torch.tensor([[True, True]]))


class Counting:
    def __init__(self): self.times = []
    def __call__(self, state, candidates, x_t, t, atom_mask):
        self.times.append(t.clone())
        scale = torch.arange(1, x_t.shape[1] + 1, device=x_t.device)[None, :, None, None, None]
        return SimpleNamespace(velocity=torch.ones_like(x_t) * scale)


def test_v3_sampler_uses_steps_minus_one_and_preserves_candidates():
    model = Counting()
    initial = torch.zeros(1, 2, 5, 7, 3)
    output = sample_rectified_flow_v3(
        model, _state(), _candidates(), torch.ones(1, 2, 5, 7, dtype=torch.bool),
        steps=9, initial_coordinates=initial,
    )
    assert output.model_calls == len(model.times) == 8
    assert model.times[-1].item() == pytest.approx(7 / 8)
    torch.testing.assert_close(output.coordinates[:, 1], output.coordinates[:, 0] * 2)
