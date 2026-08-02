from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.structural_tokens_lasso import (
    ROLE_ACCEPTOR_CARBOXYL, ROLE_BACKBONE, ROLE_NTERM_REACTIVE, ROLE_PLUG,
    ROLE_RING_CONTEXT, ROLE_SIDECHAIN, LassoStructuralTokenExpander,
)
from tests.helpers_v3 import tiny_candidate_batch


def test_required_roles_parent_mapping_and_padding():
    batch = tiny_candidate_batch(candidates_count=2, length=6)
    expander = LassoStructuralTokenExpander(8, 6)
    s = torch.randn(1, 2, 6, 8)
    z = torch.randn(1, 2, 6, 6, 6)
    state = expander(s, z, batch["token_mask"], batch["candidates"])
    assert set(state.role_id.unique().tolist()) >= {
        ROLE_BACKBONE, ROLE_SIDECHAIN, ROLE_NTERM_REACTIVE,
        ROLE_ACCEPTOR_CARBOXYL, ROLE_PLUG, ROLE_RING_CONTEXT,
    }
    assert state.parent_residue[state.role_id == ROLE_NTERM_REACTIVE].eq(0).all()
    for m in range(2):
        assert state.parent_residue[0, m][state.role_id[0, m] == ROLE_ACCEPTOR_CARBOXYL].item() == 2
        assert state.parent_residue[0, m][state.role_id[0, m] == ROLE_PLUG].item() == batch["candidates"].p[0, m]
        assert state.parent_residue[0, m][state.role_id[0, m] == ROLE_RING_CONTEXT].item() == batch["candidates"].k[0, m]


def test_role_embeddings_materially_change_collapsed_output():
    batch = tiny_candidate_batch(candidates_count=2, length=6)
    expander = LassoStructuralTokenExpander(8, 6)
    s = torch.randn(1, 2, 6, 8)
    z = torch.randn(1, 2, 6, 6, 6)
    baseline = expander(s, z, batch["token_mask"], batch["candidates"])
    collapsed_before = expander.collapse_to_residue(baseline, 6)
    with torch.no_grad():
        expander.role_single.zero_()
        expander.role_pair.weight.zero_()
    ablated = expander.collapse_to_residue(
        expander(s, z, batch["token_mask"], batch["candidates"]), 6
    )
    assert (collapsed_before[0] - ablated[0]).norm() / collapsed_before[0].norm() > 1e-3
    assert (collapsed_before[1] - ablated[1]).norm() / collapsed_before[1].norm() > 1e-3
