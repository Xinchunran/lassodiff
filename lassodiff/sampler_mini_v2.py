"""Torsion sampler with the same angular velocity convention as training."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .torsion_flow import wrap_angle
from .torsion_state import TorsionState, TorsionVelocity


@dataclass(frozen=True)
class MiniInferenceConfig:
    prior_mode: str
    projection: bool = False
    topology_seeded: bool = False
    method: str = "heun"
    steps: int = 60

    @classmethod
    def unassisted(cls, steps: int = 60):
        return cls("open_chain", False, False, "heun", steps)

    @classmethod
    def assisted(cls, steps: int = 60):
        return cls("single_crossing", False, True, "heun", steps)


def open_chain_torsion_prior(candidates: list, *, generator=None, device=None) -> TorsionState:
    if not candidates:
        raise ValueError("at least one candidate is required")
    device = torch.device(device or "cpu")
    length = max(len(candidate.sequence) for candidate in candidates)
    batch = len(candidates)
    backbone = torch.randn((batch, 1, length, 3), generator=generator, device=device) * .8
    backbone[..., 2] = torch.pi
    backbone_mask = torch.zeros_like(backbone, dtype=torch.bool)
    chi = torch.zeros((batch, 1, length, 4), device=device)
    chi_mask = torch.zeros_like(chi, dtype=torch.bool)
    for index, candidate in enumerate(candidates):
        valid_length = len(candidate.sequence)
        backbone_mask[index, :, :valid_length] = True
        chi_mask[index, :, candidate.k] = True
    return TorsionState(backbone, backbone_mask, chi, chi_mask)


@torch.no_grad()
def sample_torsion_model(model, conditioner, candidates: list, aa_ids: torch.Tensor, token_mask: torch.Tensor,
                         *, steps: int = 60, method: str = "heun", generator=None, device=None) -> TorsionState:
    if any(not isinstance(candidate, type(candidates[0])) for candidate in candidates):
        raise ValueError("candidates must contain CandidateCondition objects")
    state = open_chain_torsion_prior(candidates, generator=generator, device=device or aa_ids.device)
    state = state.to(device=aa_ids.device, dtype=aa_ids.dtype if aa_ids.is_floating_point() else torch.float32)
    conditioning = conditioner(sequences=[candidate.sequence for candidate in candidates], aa_ids=aa_ids,
                                token_mask=token_mask, k=torch.tensor([candidate.k for candidate in candidates], device=aa_ids.device),
                                p=torch.tensor([candidate.p for candidate in candidates], device=aa_ids.device))
    return integrate_torsion_flow(model, state, steps, method, {"conditioning": conditioning, "token_mask": token_mask, "candidates": candidates})


def integrate_torsion_flow(model, initial_state: TorsionState, steps: int, method: str = "euler", model_kwargs: dict | None = None) -> TorsionState:
    if steps < 2 or method not in {"euler", "heun"}:
        raise ValueError("sampler requires steps >= 2 and method Euler or Heun")
    model_kwargs = dict(model_kwargs or {})
    state = initial_state.clone()
    dt = 1.0 / (steps - 1)
    for index in range(steps - 1):
        time = torch.full((state.backbone.shape[0],), index * dt, dtype=state.backbone.dtype, device=state.backbone.device)
        velocity = model(state_t=state, time=time, **model_kwargs)
        if isinstance(velocity, TorsionVelocity):
            first = velocity
        else:
            first = velocity.velocity
        if method == "heun":
            midpoint = TorsionState(wrap_angle(state.backbone + dt * first.backbone) * state.backbone_mask, state.backbone_mask,
                                    wrap_angle(state.acceptor_chi + dt * first.acceptor_chi) * state.acceptor_chi_mask, state.acceptor_chi_mask)
            second_raw = model(state_t=midpoint, time=torch.full_like(time, (index + 1) * dt), **model_kwargs)
            second = second_raw if isinstance(second_raw, TorsionVelocity) else second_raw.velocity
            vb = (first.backbone + second.backbone) / 2
            vc = (first.acceptor_chi + second.acceptor_chi) / 2
        else:
            vb, vc = first.backbone, first.acceptor_chi
        state = TorsionState(wrap_angle(state.backbone + dt * vb) * state.backbone_mask, state.backbone_mask,
                             wrap_angle(state.acceptor_chi + dt * vc) * state.acceptor_chi_mask, state.acceptor_chi_mask)
        if not bool(torch.isfinite(state.backbone).all() and torch.isfinite(state.acceptor_chi).all()):
            raise RuntimeError(f"non-finite torsion state at sampler step {index}")
    return state
