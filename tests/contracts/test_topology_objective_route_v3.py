from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.model_v3 import LassoDiffOpenDDEV3
from lassodiff.opendde_bridge.reasoner import FrozenOpenDDEReasoner
from lassodiff.opendde_bridge.schema import OpenDDEReasoningState
from lassodiff.topology_adapter import CandidateBatch
from lassodiff.topology_objective import TopologyLossWeights, topology_supervised_candidate_loss


class FrozenStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))


def test_topology_loss_reaches_required_structure_modules_but_not_gate():
    torch.manual_seed(73)
    B, M, L, A = 1, 1, 7, 7
    state = OpenDDEReasoningState(
        torch.randn(B, L, 12), torch.randn(B, L, L, 10),
        torch.ones(B, L, dtype=torch.bool), torch.arange(L)[None], ("x",),
        "hash", "commit", 1,
    )
    model = LassoDiffOpenDDEV3(
        FrozenOpenDDEReasoner(FrozenStub(), lambda _: state), 12, 10,
        c_s=16, c_z=16, c_a=24, n_heads=4, diffusion_blocks=2,
    )
    candidates = CandidateBatch(
        torch.tensor([[3]]), torch.tensor([[4]]), torch.tensor([[1]]),
        torch.tensor([[1.0]]), torch.tensor([[True]]),
    )
    target = torch.randn(B, M, L, A, 3)
    # Make the reactive target chemically valid and leave slot O2 absent.
    target[0, 0, 0, 0] = torch.tensor([-1.33, 0.0, 0.0])
    target[0, 0, 1, 4] = torch.tensor([0.0, 0.0, 0.0])
    target[0, 0, 1, 5] = torch.tensor([0.5, 0.8660254, 0.0])
    target[0, 0, 1, 1] = torch.tensor([1.0, -1.0, 0.0])
    atom_mask = torch.ones(B, M, L, A, dtype=torch.bool)
    atom_mask[..., 6] = False
    noise = torch.randn_like(target)
    time = torch.tensor([0.25])
    x_t = (1 - time[:, None, None, None, None]) * noise + time[:, None, None, None, None] * target
    output = model(state, candidates, x_t, time, atom_mask)
    loss = topology_supervised_candidate_loss(
        output.velocity, target - noise, x_t, time, target, state.token_mask,
        atom_mask, candidates, TopologyLossWeights(),
    ).total.mean()
    loss.backward()
    assert all(parameter.grad is None for parameter in model.sequence_gate.parameters())
    assert all(parameter.grad is None for parameter in model.reasoner.parameters())
    for module in (model.reasoning_adapter, model.topology_adapter, model.structural_tokens, model.diffusion):
        assert any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in module.parameters()
        )
