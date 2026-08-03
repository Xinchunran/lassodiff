"""Connect grouped conformers to the circular flow training objective."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .atom_schema_lasso import CandidateCondition
from .sampler_mini_v2 import open_chain_torsion_prior
from .torsion_flow import interpolate_torsion_state
from .torsion_state import TorsionState, TorsionVelocity


@dataclass
class PreparedMiniV2Batch:
    sequences: list[str]
    aa_ids: torch.Tensor
    token_mask: torch.Tensor
    k: torch.Tensor
    p: torch.Tensor
    candidates: list[CandidateCondition]
    target_conformer_index: torch.Tensor
    target_state: TorsionState
    source_state: TorsionState
    state_t: TorsionState
    target_velocity: TorsionVelocity
    time: torch.Tensor
    all_backbone_targets: torch.Tensor
    all_backbone_masks: torch.Tensor
    all_chi_targets: torch.Tensor
    all_chi_masks: torch.Tensor
    core_targets: torch.Tensor
    core_target_masks: torch.Tensor
    atom14_targets: torch.Tensor
    atom14_target_masks: torch.Tensor
    conformer_mask: torch.Tensor


def prepare_backbone_flow_batch(batch: dict, *, device: torch.device,
                                generator: torch.Generator, num_samples: int = 1,
                                min_time: float = .02, max_time: float = .98,
                                global_step: int | None = None) -> PreparedMiniV2Batch:
    sequences = list(batch["sequences"])
    # The deterministic one-example gate deliberately fixes its source prior
    # and time grid; production batches leave this flag absent.
    if batch.get("fixed_flow", False):
        generator = torch.Generator().manual_seed(23)
    aa_ids = batch["aa_ids"].to(device)
    token_mask = batch["token_mask"].to(device)
    k, p = batch["k"].to(device), batch["p"].to(device)
    conformer_mask = batch["conformer_mask"].to(device)
    candidates = [CandidateCondition(s, int(ki), int(pi)) for s, ki, pi in zip(sequences, k.tolist(), p.tolist())]
    B, M = conformer_mask.shape
    selected = torch.empty(B, dtype=torch.long, device=device)
    for b in range(B):
        valid = torch.nonzero(conformer_mask[b], as_tuple=False).flatten()
        if not valid.numel():
            raise RuntimeError(f"candidate {b} has no valid conformers")
        selected[b] = valid[torch.randint(valid.numel(), (1,), generator=generator).item()]
    if batch.get("fixed_flow", False):
        # Keep the deterministic gate's prior identical to the sampler's
        # seed; conformer selection must not consume prior randomness.
        generator = torch.Generator().manual_seed(23)
    bi = torch.arange(B, device=device)
    target_backbone = batch["backbone_torsions"].to(device)[bi, selected]
    target_backbone_mask = batch["backbone_torsion_masks"].to(device)[bi, selected]
    target_chi = batch["chi_targets"].to(device)[bi, selected]
    target_chi_mask = batch["chi_masks"].to(device)[bi, selected]
    def expand(x):
        return x[:, None].expand(B, num_samples, *x.shape[1:]).clone()
    target_state = TorsionState(expand(target_backbone), expand(target_backbone_mask), expand(target_chi), expand(target_chi_mask))
    source_state = open_chain_torsion_prior(candidates, num_samples=num_samples, generator=generator, device=device)
    source_mask_b = source_state.backbone_mask & target_state.backbone_mask
    source_mask_c = source_state.acceptor_chi_mask & target_state.acceptor_chi_mask
    source_state = TorsionState(source_state.backbone * source_mask_b, source_mask_b,
                                source_state.acceptor_chi * source_mask_c, source_mask_c)
    target_state = TorsionState(target_state.backbone * source_mask_b, source_mask_b,
                                target_state.acceptor_chi * source_mask_c, source_mask_c)
    if batch.get("fixed_flow", False) and global_step is not None:
        phase = (int(global_step) - 1) % 10 / 9.0
        time = torch.full(
            (B, num_samples),
            min_time + phase * (max_time - min_time),
            device=device,
        )
    else:
        time = torch.rand((B, num_samples), generator=generator, device=device) * (max_time - min_time) + min_time
    flow = interpolate_torsion_state(source_state, target_state, time)
    return PreparedMiniV2Batch(
        sequences, aa_ids, token_mask, k, p, candidates, selected, target_state, source_state,
        flow.state_t, flow.velocity, time,
        batch["backbone_torsions"].to(device), batch["backbone_torsion_masks"].to(device),
        batch["chi_targets"].to(device), batch["chi_masks"].to(device),
        batch["core_targets"].to(device), batch["core_target_masks"].to(device),
        batch["atom14_targets"].to(device), batch["atom14_target_masks"].to(device), conformer_mask,
    )
