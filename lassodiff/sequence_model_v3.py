from __future__ import annotations

import torch.nn as nn

from .reasoning_adapter import OpenDDEReasoningAdapter
from .sequence_gate import LassoSequenceGate


class CachedOpenDDESequenceModel(nn.Module):
    """Trainable adapter/gate over immutable cached OpenDDE states."""

    def __init__(self, c_s_open=384, c_z_open=384, c_s=384, c_z=192, max_candidates=3):
        super().__init__()
        self.reasoning_adapter = OpenDDEReasoningAdapter(c_s_open, c_z_open, c_s, c_z)
        self.sequence_gate = LassoSequenceGate(c_s, c_z, c_s, max_candidates)

    def forward(self, state):
        single, pair = self.reasoning_adapter(state.single, state.pair, state.token_mask)
        return self.sequence_gate(single, pair, state.token_mask, state.residue_type, state.residue_index)
