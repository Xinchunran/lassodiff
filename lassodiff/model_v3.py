from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .opendde_bridge.schema import OpenDDEReasoningState, validate_reasoning_state
from .reasoning_adapter import OpenDDEReasoningAdapter, scale_gradient
from .score_head_lasso_v3 import EquivariantLassoDiffusionV3
from .sequence_gate import LassoSequenceGate, SequenceAssessment, decide_sequence
from .structural_tokens_lasso import LassoStructuralTokenExpander, StructuralTokenState
from .topology_adapter import CandidateBatch, LassoTopologyAdapterV3


ARCHITECTURE_ID_V3 = "lassodiff_opendde_v3"
SCHEMA_VERSION_V3 = 3


@dataclass
class V3ArchitectureTrace:
    reasoning_backend: str = "opendde_pretrained"
    reasoning_checkpoint_sha256: str = ""
    reasoner_called: int = 0
    reasoning_adapter_called: int = 0
    sequence_gate_called: int = 0
    topology_adapter_calls: int = 0
    structural_token_calls: int = 0
    geometry_calls: int = 0
    diffusion_calls: int = 0
    candidate_count: int = 0
    projection_used: bool = False
    topology_guidance_used: bool = False


@dataclass
class V3StructureOutput:
    velocity: torch.Tensor
    candidate_mask: torch.Tensor
    structural_tokens: StructuralTokenState
    trace: V3ArchitectureTrace | None


class LassoDiffOpenDDEV3(nn.Module):
    architecture_id = ARCHITECTURE_ID_V3
    schema_version = SCHEMA_VERSION_V3

    def __init__(
        self, reasoner: nn.Module, c_s_open: int, c_z_open: int, *,
        c_s: int = 384, c_z: int = 192, c_a: int = 256, n_heads: int = 8,
        diffusion_blocks: int = 8, max_candidates: int = 3, structure_gradient_scale: float = 0.1,
    ):
        super().__init__()
        self.reasoner = reasoner
        self.reasoning_adapter = OpenDDEReasoningAdapter(c_s_open, c_z_open, c_s, c_z)
        self.sequence_gate = LassoSequenceGate(c_s, c_z, hidden_dim=c_s, max_candidates=max_candidates)
        self.topology_adapter = LassoTopologyAdapterV3(c_s, c_z)
        self.structural_tokens = LassoStructuralTokenExpander(c_s, c_z)
        self.diffusion = EquivariantLassoDiffusionV3(c_s, c_z, c_a, diffusion_blocks, n_heads)
        self.structure_gradient_scale = float(structure_gradient_scale)

    def required_trainable_modules(self):
        return [self.reasoning_adapter, self.sequence_gate, self.topology_adapter, self.structural_tokens, self.diffusion]

    def forward_reasoning(self, features, trace: V3ArchitectureTrace | None = None):
        state = self.reasoner(features)
        if trace is not None:
            trace.reasoner_called += 1
            trace.reasoning_checkpoint_sha256 = state.checkpoint_sha256
        return state

    def _adapt(self, state: OpenDDEReasoningState, trace=None):
        state = validate_reasoning_state(state)
        single, pair = self.reasoning_adapter(state.single, state.pair, state.token_mask)
        if trace is not None:
            trace.reasoning_adapter_called += 1
            trace.reasoning_checkpoint_sha256 = state.checkpoint_sha256
        return single, pair

    def assess_sequence(self, state: OpenDDEReasoningState, trace: V3ArchitectureTrace | None = None) -> SequenceAssessment:
        single, pair = self._adapt(state, trace)
        assessment = self.sequence_gate(
            single, pair, state.token_mask, residue_type=state.residue_type,
            residue_index=state.residue_index,
        )
        if trace is not None:
            trace.sequence_gate_called += 1
        return assessment

    @staticmethod
    def structure_prior(assessment: SequenceAssessment, candidate_mask: torch.Tensor):
        M = candidate_mask.shape[1]
        logits = assessment.candidate_logits[:, :M].masked_fill(~candidate_mask, -torch.inf)
        return torch.softmax(logits, dim=-1).detach()

    def predict_velocity(
        self, state: OpenDDEReasoningState, candidates: CandidateBatch, x_t, t, atom_mask,
        *, return_trace: bool = False,
    ) -> V3StructureOutput:
        trace = V3ArchitectureTrace() if return_trace else None
        single, pair = self._adapt(state, trace)
        single = scale_gradient(single, self.structure_gradient_scale)
        pair = scale_gradient(pair, self.structure_gradient_scale)
        candidate_single, candidate_pair, candidate_pair_mask = self.topology_adapter(
            single, pair, state.token_mask, candidates
        )
        if trace is not None:
            trace.topology_adapter_calls += 1
            trace.candidate_count = int(candidates.candidate_mask.sum().item())
        structural = self.structural_tokens(candidate_single, candidate_pair, state.token_mask, candidates)
        if trace is not None:
            trace.structural_token_calls += 1
        residue_single, residue_pair = self.structural_tokens.collapse_to_residue(
            structural, state.token_mask.shape[1]
        )
        velocity = self.diffusion(
            x_t, t, residue_single, residue_pair, state.token_mask, atom_mask,
            candidate_pair_mask, trace=trace, candidates=candidates,
        )
        return V3StructureOutput(velocity, candidates.candidate_mask, structural, trace)

    def forward(
        self, state, candidates=None, x_t=None, t=None, atom_mask=None, *, return_trace: bool = False,
    ):
        if candidates is None:
            if any(value is not None for value in (x_t, t, atom_mask)):
                raise ValueError("sequence-only V3 forward cannot receive structure tensors")
            return self.assess_sequence(state)
        return self.predict_velocity(
            state, candidates, x_t, t, atom_mask, return_trace=return_trace,
        )

    def screen_state(
        self, state: OpenDDEReasoningState, *, reject_threshold: float,
        accept_threshold: float, ood_threshold: float,
    ):
        assessment = self.assess_sequence(state)
        lasso_probability = 1.0 - torch.sigmoid(assessment.no_lasso_logit)
        return {
            "decision": decide_sequence(
                lasso_probability, assessment.ood_score, reject_threshold=reject_threshold,
                accept_threshold=accept_threshold, ood_threshold=ood_threshold,
            ),
            "lasso_probability": lasso_probability,
            "assessment": assessment,
            "projection_used": False,
            "topology_guidance_used": False,
        }
