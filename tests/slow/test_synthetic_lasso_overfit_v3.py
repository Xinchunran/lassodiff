from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.model_v3 import LassoDiffOpenDDEV3
from lassodiff.opendde_bridge.reasoner import CacheOnlyOpenDDEReasoner
from lassodiff.opendde_bridge.schema import OpenDDEReasoningState
from lassodiff.topology_adapter import CandidateBatch


def test_tiny_v3_can_reduce_gate_and_candidate_velocity_error():
    torch.manual_seed(41)
    B, L, M, A = 2, 5, 1, 7
    state = OpenDDEReasoningState(
        torch.randn(B, L, 9), torch.randn(B, L, L, 7), torch.ones(B, L, dtype=torch.bool),
        torch.arange(L)[None].expand(B, -1), ("p", "n"), "h", "c", 1,
        residue_type=torch.tensor([[0,2,4,5,6],[0,3,4,5,6]]),
    )
    candidates = CandidateBatch(torch.ones(B,M,dtype=torch.long), torch.full((B,M),3),
        torch.ones(B,M,dtype=torch.long), torch.ones(B,M), torch.ones(B,M,dtype=torch.bool))
    model = LassoDiffOpenDDEV3(CacheOnlyOpenDDEReasoner("h","c"), 9, 7, c_s=12, c_z=12,
                               c_a=16, n_heads=4, diffusion_blocks=1, max_candidates=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = torch.randn(B,M,L,A,3); target = x * .15
    atoms = torch.ones(B,M,L,A,dtype=torch.bool); time = torch.full((B,),.5)

    def loss():
        assessment = model.assess_sequence(state)
        gate_target = torch.tensor([0.,1.])  # H0 target: positive then negative.
        gate = torch.nn.functional.binary_cross_entropy_with_logits(assessment.no_lasso_logit, gate_target)
        velocity = model(state,candidates,x,time,atoms).velocity
        return gate + (velocity-target).square().mean(), gate, (velocity-target).square().mean()

    initial = tuple(float(value) for value in loss())
    for _ in range(80):
        optimizer.zero_grad(); total, _gate, _structure = loss(); total.backward(); optimizer.step()
    final = tuple(float(value) for value in loss())
    assert final[0] < initial[0] * .35
    assert final[1] < initial[1] * .35
    assert final[2] < initial[2] * .35
