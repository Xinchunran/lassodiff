"""Projection-free construction/screening sampler initialized from peptide priors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .atom_schema_lasso import CandidateCondition
from .peptide_prior import sample_peptide_prior
from .seq_encoder import seq_to_aa_ids
from .topology_adapter import CandidateBatch


InferenceMode = Literal["construction", "screening"]


@dataclass(frozen=True)
class MiniSamplerOutput:
    coordinates: torch.Tensor
    prior_mode: str
    model_calls: int
    projection_used: bool
    topology_guidance_used: bool


@torch.no_grad()
def sample_mini(
    model,
    candidate: CandidateCondition,
    *,
    inference_mode: InferenceMode,
    steps: int = 40,
    generator: torch.Generator | None = None,
    prior_mode: str | None = None,
    device: torch.device | str | None = None,
):
    if steps < 2:
        raise ValueError("Mini sampler requires at least two time points")
    if inference_mode == "screening":
        if prior_mode not in (None, "open_chain"):
            raise ValueError("screening only permits the open-chain prior")
        prior_mode = "open_chain"
    elif inference_mode == "construction":
        prior_mode = prior_mode or "single_crossing"
    else:
        raise ValueError("inference_mode must be construction or screening")
    prior = sample_peptide_prior(candidate, mode=prior_mode, generator=generator)
    parameter = next(model.parameters())
    device = torch.device(device) if device is not None else parameter.device
    x = prior.coordinates.to(device)[None, None]
    atom_mask = prior.atom_mask.to(device)[None, None]
    length = len(candidate.sequence)
    aa_ids = seq_to_aa_ids(candidate.sequence).to(device)[None]
    token_mask = torch.ones((1, length), dtype=torch.bool, device=device)
    candidates = CandidateBatch(
        torch.tensor([[candidate.k]], device=device), torch.tensor([[candidate.p]], device=device),
        torch.tensor([[candidate.k]], device=device), torch.ones((1, 1), device=device),
        torch.ones((1, 1), dtype=torch.bool, device=device),
        acceptor_type=torch.tensor([[candidate.sequence[candidate.k] == "E"]], device=device),
    )
    dt = 1.0 / (steps - 1)
    for index in range(steps - 1):
        t = torch.tensor([index * dt], dtype=x.dtype, device=device)
        output = model(aa_ids, token_mask, candidates, x, t, atom_mask)
        x = (x + dt * output.velocity) * atom_mask[..., None]
        if not bool(torch.isfinite(x).all()):
            raise RuntimeError(f"Mini sampler produced non-finite coordinates at update {index}")
    return MiniSamplerOutput(x[0, 0], prior_mode, steps - 1, False, False)
