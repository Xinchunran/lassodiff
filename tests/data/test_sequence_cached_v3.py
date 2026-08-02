from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lassodiff.data.sequence_cached import collate_sequence_gate
from lassodiff.data.sequence_v3 import SequenceExample
from lassodiff.opendde_bridge.schema import OpenDDEReasoningState


def _state(length):
    return OpenDDEReasoningState(torch.randn(1,length,4), torch.randn(1,length,length,3),
        torch.ones(1,length,dtype=torch.bool), torch.arange(length)[None], (str(length),), "h", "c", 1)


def test_sequence_collate_preserves_nonuniform_teacher_prior_and_padding():
    examples = [
        SequenceExample("p", "ACDEF", 1, "positive", "verified_lasso", "a", 2, 4, False, [.8,.2]),
        SequenceExample("n", "ACDEFGG", 0, "background", "background_non_lasso", "b", ood=True),
    ]
    batch = collate_sequence_gate([(examples[0], _state(5)), (examples[1], _state(7))])
    torch.testing.assert_close(batch["teacher_prior"][0], torch.tensor([.8,.2,0.]))
    assert batch["candidate_mask"].tolist() == [[True,True,False],[False,False,False]]
    assert batch["reasoning_state"].token_mask[0].tolist() == [True]*5+[False]*2
