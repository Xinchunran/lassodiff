"""Runtime route/gradient/optimizer verifier for LassoDiff OpenDDE V3."""
from __future__ import annotations

from dataclasses import replace
import inspect

import torch

from .architecture_contract_v3 import V3ArchitectureConfig


FORBIDDEN_GATE_INPUTS = {
    "k", "p", "lasso_feats", "is_ring", "is_loop", "is_tail",
    "acceptor_index", "candidate_rank", "closure_edge",
}


def _trainable(module):
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def _nonzero_grad(module):
    return any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        and float(parameter.grad.detach().abs().sum()) > 0
        for parameter in module.parameters() if parameter.requires_grad
    )


def _zero_grad(module):
    return all(parameter.grad is None or float(parameter.grad.detach().abs().sum()) == 0 for parameter in module.parameters())


def _optimizer_parameter_ids(optimizer):
    ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("optimizer contains duplicate parameters")
    return set(ids)


def build_v3_optimizer(model, config: V3ArchitectureConfig):
    groups = [
        {"name": "reasoning_adapter", "params": _trainable(model.reasoning_adapter), "lr": config.training.adapter_lr},
        {"name": "sequence_gate", "params": _trainable(model.sequence_gate), "lr": config.training.gate_lr},
        {"name": "topology_adapter", "params": _trainable(model.topology_adapter), "lr": config.training.structure_lr},
        {"name": "structural_tokens", "params": _trainable(model.structural_tokens), "lr": config.training.structure_lr},
        {"name": "diffusion", "params": _trainable(model.diffusion), "lr": config.training.structure_lr},
    ]
    if any(not group["params"] for group in groups):
        raise RuntimeError("required V3 optimizer group has no trainable parameters")
    return torch.optim.AdamW(groups, weight_decay=0.01)


def verify_model_contract_v3(
    model, state, candidates, optimizer, config: V3ArchitectureConfig, *, reasoner_features=None,
):
    config.validate()
    trainable_numel = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_numel != config.model.expected_trainable_numel:
        raise RuntimeError(
            f"V3 trainable parameter manifest mismatch: expected={config.model.expected_trainable_numel}, got={trainable_numel}"
        )
    if reasoner_features is None:
        raise RuntimeError("V3 preflight requires real/stub reasoner features for route verification")
    from .model_v3 import V3ArchitectureTrace
    reasoner_trace = V3ArchitectureTrace()
    routed_state = model.forward_reasoning(reasoner_features, reasoner_trace)
    if reasoner_trace.reasoner_called != 1:
        raise RuntimeError("OpenDDE reasoner route was not called exactly once")
    if state is not None and routed_state.sequence_hashes != state.sequence_hashes:
        raise RuntimeError("reasoner route returned a different sequence state")
    state = routed_state
    forbidden = set(inspect.signature(model.assess_sequence).parameters) & FORBIDDEN_GATE_INPUTS
    forbidden |= set(inspect.signature(model.sequence_gate.forward).parameters) & FORBIDDEN_GATE_INPUTS
    if forbidden:
        raise RuntimeError(f"sequence gate exposes forbidden topology inputs: {sorted(forbidden)}")
    model.train()
    if model.reasoner.model.training or any(parameter.requires_grad for parameter in model.reasoner.parameters()):
        raise RuntimeError("OpenDDE reasoner is not frozen/eval")
    if state.checkpoint_sha256 != config.reasoning.checkpoint_sha256:
        raise RuntimeError("reasoning state checkpoint provenance mismatch")
    if state.opendde_commit != config.reasoning.opendde_commit:
        raise RuntimeError("reasoning state commit provenance mismatch")
    if state.feature_schema_version != config.reasoning.cache_schema_version:
        raise RuntimeError("reasoning state feature schema mismatch")

    required = {id(parameter) for module in model.required_trainable_modules() for parameter in _trainable(module)}
    optimizer_ids = _optimizer_parameter_ids(optimizer)
    missing = required - optimizer_ids
    if missing:
        raise RuntimeError(f"optimizer missing {len(missing)} required trainable parameters")

    model.zero_grad(set_to_none=True)
    sequence_trace = V3ArchitectureTrace()
    assessment = model.assess_sequence(state, sequence_trace)
    acceptor_valid = assessment.acceptor_logits > -1e20
    plug_valid = assessment.plug_logits > -1e20
    if not acceptor_valid.any() or not plug_valid.any():
        raise RuntimeError("preflight sequence lacks legal acceptor/plug positions")
    gate_loss = (
        assessment.no_lasso_logit.square().mean() + assessment.acceptor_logits[acceptor_valid].square().mean()
        + assessment.plug_logits[plug_valid].square().mean() + assessment.candidate_logits.square().mean()
        + assessment.ood_score.square().mean()
    )
    gate_loss.backward()
    if not _nonzero_grad(model.reasoning_adapter) or not _nonzero_grad(model.sequence_gate):
        raise RuntimeError("gate route has no finite nonzero gradient")
    if not all(_zero_grad(module) for module in (model.topology_adapter, model.structural_tokens, model.diffusion)):
        raise RuntimeError("gate loss leaked gradient into structure generator")

    with torch.no_grad():
        base_gate = model.assess_sequence(state).no_lasso_logit
        pair_zero = replace(state, pair=torch.zeros_like(state.pair))
        single_zero = replace(state, single=torch.zeros_like(state.single))
        pair_gate = model.assess_sequence(pair_zero).no_lasso_logit
        single_gate = model.assess_sequence(single_zero).no_lasso_logit
        gate_denominator = base_gate.norm().clamp_min(1e-8)
        pair_gate_change = (base_gate - pair_gate).norm() / gate_denominator
        single_gate_change = (base_gate - single_gate).norm() / gate_denominator
    if pair_gate_change <= 1e-6 or single_gate_change <= 1e-6:
        raise RuntimeError("sequence output does not depend on OpenDDE single/pair state")

    B, L = state.token_mask.shape
    M = candidates.k.shape[1]
    A = 7
    generator = torch.Generator(device=state.single.device).manual_seed(1701)
    x_t = torch.randn((B, M, L, A, 3), generator=generator, device=state.single.device)
    atom_mask = state.token_mask[:, None, :, None].expand(B, M, L, A).clone()
    t = torch.full((B,), 0.5, device=state.single.device)
    model.zero_grad(set_to_none=True)
    structure = model.predict_velocity(state, candidates, x_t, t, atom_mask, return_trace=True)
    structure.velocity.square().mean().backward()
    if any(not _nonzero_grad(module) for module in (
        model.reasoning_adapter, model.topology_adapter, model.structural_tokens, model.diffusion
    )):
        raise RuntimeError("structure route has a required module without finite nonzero gradient")
    if not _zero_grad(model.sequence_gate) or not _zero_grad(model.reasoner):
        raise RuntimeError("structure loss leaked into sequence gate or frozen reasoner")
    trace = structure.trace
    trace.reasoner_called = reasoner_trace.reasoner_called
    trace.sequence_gate_called = sequence_trace.sequence_gate_called
    trace.reasoning_adapter_called += sequence_trace.reasoning_adapter_called
    if trace.topology_adapter_calls != 1 or trace.structural_token_calls != 1:
        raise RuntimeError("V3 topology/structural route is incomplete")
    if trace.geometry_calls != config.model.diffusion_blocks or trace.diffusion_calls != config.model.diffusion_blocks:
        raise RuntimeError("dynamic geometry was not recomputed in every diffusion block")

    with torch.no_grad():
        pair_structure = model.predict_velocity(pair_zero, candidates, x_t, t, atom_mask).velocity
        structure_change = (structure.velocity.detach() - pair_structure).norm() / structure.velocity.detach().norm().clamp_min(1e-8)
    if structure_change <= 1e-6:
        raise RuntimeError("structure output does not depend on OpenDDE pair state")
    screening = model.screen_state(
        state, reject_threshold=config.sequence_gate.reject_threshold,
        accept_threshold=config.sequence_gate.accept_threshold,
        ood_threshold=config.sequence_gate.ood_threshold,
    )
    if screening["projection_used"] or screening["topology_guidance_used"]:
        raise RuntimeError("screening used forbidden projection/guidance")
    return {
        "status": "PASS", "architecture_id": model.architecture_id,
        "sequence_single_relative_change": float(single_gate_change),
        "sequence_pair_relative_change": float(pair_gate_change),
        "structure_pair_relative_change": float(structure_change),
        "optimizer_parameter_count": len(optimizer_ids), "trainable_parameter_numel": trainable_numel,
        "trace": trace.__dict__,
        "reasoning_state": {
            "single_shape": list(state.single.shape), "pair_shape": list(state.pair.shape),
            "feature_schema_version": state.feature_schema_version,
            "checkpoint_sha256": state.checkpoint_sha256, "opendde_commit": state.opendde_commit,
        },
    }
