"""Torsion sampler with the same angular velocity convention as training."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .torsion_flow import wrap_angle
from .torsion_state import TorsionState, TorsionVelocity


@dataclass
class MiniV2SampleOutput:
    state: TorsionState
    core_coordinates: torch.Tensor
    atom14_coordinates: torch.Tensor | None
    atom14_mask: torch.Tensor | None
    finite: torch.Tensor


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


def open_chain_torsion_prior(candidates: list, *, num_samples: int = 1, generator=None, device=None) -> TorsionState:
    if not candidates:
        raise ValueError("at least one candidate is required")
    device = torch.device(device or "cpu")
    length = max(len(candidate.sequence) for candidate in candidates)
    batch = len(candidates)
    backbone = torch.randn((batch, num_samples, length, 3), generator=generator, device=device) * .8
    backbone[..., 2] = torch.pi
    backbone_mask = torch.zeros_like(backbone, dtype=torch.bool)
    chi = torch.zeros((batch, num_samples, length, 4), device=device)
    chi_mask = torch.zeros_like(chi, dtype=torch.bool)
    for index, candidate in enumerate(candidates):
        valid_length = len(candidate.sequence)
        backbone_mask[index, :, :valid_length, :] = True
        backbone_mask[index, :, 0, 0] = False
        backbone_mask[index, :, valid_length - 1, 1:] = False
        n_chi = 2 if candidate.sequence[candidate.k] == "D" else 3
        chi_mask[index, :, candidate.k, :n_chi] = True
    return TorsionState(backbone, backbone_mask, chi, chi_mask)


@torch.no_grad()
def sample_torsion_model(model, conditioner, candidates: list, aa_ids: torch.Tensor, token_mask: torch.Tensor,
                         *, steps: int = 60, method: str = "heun", generator=None, device=None,
                         config: MiniInferenceConfig | None = None, num_samples: int = 1,
                         sidechain_head=None, refiner=None) -> MiniV2SampleOutput:
    if any(not isinstance(candidate, type(candidates[0])) for candidate in candidates):
        raise ValueError("candidates must contain CandidateCondition objects")
    config = config or MiniInferenceConfig.unassisted(steps=steps)
    if config.prior_mode == "open_chain":
        state = open_chain_torsion_prior(candidates, num_samples=num_samples, generator=generator, device=device or aa_ids.device)
    elif config.prior_mode == "single_crossing":
        state = assisted_lasso_torsion_prior(candidates, num_samples=num_samples, generator=generator, device=device or aa_ids.device)
    else:
        raise ValueError(f"unknown prior mode {config.prior_mode}")
    state = state.to(device=aa_ids.device, dtype=torch.float32)
    conditioning = conditioner(sequences=[candidate.sequence for candidate in candidates], aa_ids=aa_ids,
                                token_mask=token_mask, k=torch.tensor([candidate.k for candidate in candidates], device=aa_ids.device),
                                p=torch.tensor([candidate.p for candidate in candidates], device=aa_ids.device))
    final = integrate_torsion_flow(model, state, config.steps, config.method,
                                   {"conditioning": conditioning, "token_mask": token_mask, "candidates": candidates})
    from .lasso_core_decoder import decode_lasso_core
    from .chi_geometry import build_atom14_from_rigid_groups
    from .residue_constants_mini import CHI_ATOMS
    core = decode_lasso_core(final, sequences=[x.sequence for x in candidates], candidates=candidates, token_mask=token_mask)
    finite = torch.isfinite(core).all(dim=(-1, -2, -3))
    atom14 = core.new_zeros((len(candidates), final.backbone.shape[1], core.shape[2], 14, 3))
    atom14_mask = torch.zeros(atom14.shape[:-1], dtype=torch.bool, device=core.device)
    if sidechain_head is not None:
        predicted_chi = sidechain_head(conditioning.single)
    else:
        predicted_chi = core.new_zeros((len(candidates), core.shape[2], 4))
    for b, candidate in enumerate(candidates):
        aa = aa_ids[b, :len(candidate.sequence)]
        chi_mask = torch.zeros((len(candidate.sequence), 4), dtype=torch.bool, device=core.device)
        for residue, amino_acid in enumerate(candidate.sequence):
            chi_mask[residue, :len(CHI_ATOMS.get(amino_acid, ()))] = True
        for sample in range(final.backbone.shape[1]):
            chi = predicted_chi[b, :len(candidate.sequence)].clone()
            chi[candidate.k] = final.acceptor_chi[b, sample, candidate.k]
            chi_mask[candidate.k] = final.acceptor_chi_mask[b, sample, candidate.k]
            coords, mask = build_atom14_from_rigid_groups(
                core[b, sample, :len(candidate.sequence)], aa,
                chi, chi_mask, candidate,
            )
            atom14[b, sample, :len(candidate.sequence)] = coords
            atom14_mask[b, sample, :len(candidate.sequence)] = mask
    if refiner is not None:
        from .covalent_graph import build_atom14_covalent_graph
        B, Ns, L = atom14.shape[:3]
        graph = build_atom14_covalent_graph(aa_ids, token_mask, candidates)
        flat_coordinates = atom14.reshape(B * Ns, L, 14, 3)
        flat_mask = atom14_mask.reshape(B * Ns, L, 14)
        flat_aa = aa_ids[:, None].expand(B, Ns, L).reshape(B * Ns, L)
        adjacency = graph.adjacency[:, None].expand(B, Ns, L * 14, L * 14).reshape(B * Ns, L * 14, L * 14)
        bond_type = graph.bond_type[:, None].expand(B, Ns, L * 14, L * 14).reshape(B * Ns, L * 14, L * 14)
        atom14 = refiner(
            flat_coordinates, flat_aa, flat_mask,
            covalent_adjacency=adjacency, bond_type=bond_type,
        ).reshape(B, Ns, L, 14, 3)
    finite = finite & torch.isfinite(atom14).all(dim=(-1, -2, -3))
    return MiniV2SampleOutput(final, core, atom14, atom14_mask, finite)


def assisted_lasso_torsion_prior(candidates: list, *, num_samples: int = 1, generator=None, device=None) -> TorsionState:
    """Explicit secondary prior; never used by the unassisted path."""
    state = open_chain_torsion_prior(candidates, num_samples=num_samples, generator=generator, device=device)
    for b, candidate in enumerate(candidates):
        state.backbone[b, :, :, 1] = 0.0
        state.backbone[b, :, :candidate.k + 1, 0] = 0.5
    return state


def integrate_torsion_flow(model, initial_state: TorsionState, steps: int, method: str = "euler", model_kwargs: dict | None = None) -> TorsionState:
    if steps < 2 or method not in {"euler", "heun"}:
        raise ValueError("sampler requires steps >= 2 and method Euler or Heun")
    model_kwargs = dict(model_kwargs or {})
    state = initial_state.clone()
    dt = 1.0 / (steps - 1)
    for index in range(steps - 1):
        time = torch.full(state.backbone.shape[:2], index * dt, dtype=state.backbone.dtype, device=state.backbone.device)
        velocity = model(state_t=state, time=time, **model_kwargs)
        if isinstance(velocity, TorsionVelocity):
            first = velocity
        else:
            first = velocity.velocity
        # At t=1 different random-source paths share an endpoint but need not
        # share a velocity.  Querying the model there makes the final Heun
        # correction ill-posed.  Use the last identifiable t<1 velocity for
        # the endpoint step (equivalent to the reference Euler denoise).
        if method == "heun" and index < steps - 2:
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
