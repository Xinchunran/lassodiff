"""Unified Mini training graph suitable for a single FSDP root."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .atom_refiner import MiniAtomRefiner
from .candidate_viability import CandidateViabilityHead
from .flow_matching import align_target_to_source, flow_interpolate
from .losses_mini import mini_core_loss
from .model_mini import MiniCoreDiffusion
from .sidechain_builder import RotamerChiHead, build_atom14, symmetry_aware_coordinate_loss
from .topology_adapter import CandidateBatch


@dataclass
class MiniTrainingOutput:
    total: torch.Tensor
    core: torch.Tensor
    sidechain: torch.Tensor
    refine: torch.Tensor
    viability: torch.Tensor


class MiniTrainingSystem(nn.Module):
    """All trainable modules execute under one forward/FSDP boundary."""

    def __init__(self, hidden_dim: int = 128, diffusion_blocks: int = 4):
        super().__init__()
        self.model = MiniCoreDiffusion(hidden_dim=hidden_dim, blocks=diffusion_blocks)
        self.sidechain = RotamerChiHead(hidden_dim)
        self.refiner = MiniAtomRefiner()
        self.viability = CandidateViabilityHead()

    def forward(
        self, *, aa_ids, token_mask, target, core_mask, target14, target14_mask,
        k, p, acceptor_type, conditions, x0, t,
    ) -> MiniTrainingOutput:
        B, _M, L, _A, _ = target.shape
        candidates = CandidateBatch(
            k, p, k, torch.ones((B, 1), device=target.device),
            torch.ones((B, 1), dtype=torch.bool, device=target.device),
            acceptor_type=acceptor_type,
        )
        x_t, target_velocity, aligned_core = flow_interpolate(x0, target, t, core_mask)
        output = self.model(aa_ids, token_mask, candidates, x_t, t, core_mask)
        core_loss = mini_core_loss(
            output.velocity, target_velocity, x_t, t, aligned_core,
            token_mask, core_mask, candidates,
        )
        endpoint = x_t + (1 - t[:, None, None, None, None]) * output.velocity
        side_prediction = self.sidechain(output.residue_representation[:, 0])
        predicted_atom14, predicted_masks = [], []
        for b, condition in enumerate(conditions):
            length = len(condition.sequence)
            built, mask = build_atom14(
                endpoint[b, 0, :length], condition,
                side_prediction.chi_sin_cos[b, :length],
            )
            padded = endpoint.new_zeros((L, 14, 3))
            padded[:length] = built
            padded_mask = torch.zeros((L, 14), dtype=torch.bool, device=target.device)
            padded_mask[:length] = mask
            predicted_atom14.append(padded)
            predicted_masks.append(padded_mask)
        predicted_atom14 = torch.stack(predicted_atom14)
        predicted_masks = torch.stack(predicted_masks)
        target14_mask = target14_mask & predicted_masks
        aligned_target14 = align_target_to_source(predicted_atom14, target14, target14_mask)
        side_loss = endpoint.sum() * 0
        for b, condition in enumerate(conditions):
            length = len(condition.sequence)
            side_loss = side_loss + symmetry_aware_coordinate_loss(
                predicted_atom14[b, :length], aligned_target14[b, :length],
                condition.sequence, target14_mask[b, :length], condition,
            )
        side_loss = side_loss / B
        rotamer_angle = torch.atan2(
            side_prediction.chi_sin_cos[..., 0, 0], side_prediction.chi_sin_cos[..., 0, 1],
        )
        rotamer_target = torch.floor(
            (rotamer_angle.detach() + torch.pi) / (2 * torch.pi / 3),
        ).long().clamp(0, 2)
        rotamer_mask = token_mask & aa_ids.ne(5)
        rotamer_loss = torch.nn.functional.cross_entropy(
            side_prediction.rotamer_logits[rotamer_mask], rotamer_target[rotamer_mask],
        )
        side_loss = side_loss + .05 * rotamer_loss
        refined = self.refiner(predicted_atom14, aa_ids, predicted_masks)
        refine_loss = (
            ((refined - aligned_target14).square().sum(-1) * target14_mask).sum()
            / target14_mask.sum().clamp_min(1)
        )
        negative_p = torch.where(
            p + 1 < token_mask.sum(1, keepdim=True) - 1, p + 1, p - 1,
        )
        viability_k = torch.cat((k, k), dim=1)
        viability_p = torch.cat((p, negative_p), dim=1)
        viability_logits = self.viability(aa_ids, token_mask, viability_k, viability_p)
        correct_logit, negative_logit = viability_logits[:, :1], viability_logits[:, 1:]
        viability_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            correct_logit, torch.ones_like(correct_logit),
        )
        finite_negative = torch.isfinite(negative_logit)
        if bool(finite_negative.any()):
            viability_loss = viability_loss + torch.nn.functional.binary_cross_entropy_with_logits(
                negative_logit[finite_negative], torch.zeros_like(negative_logit[finite_negative]),
            )
        total = core_loss.total + .2 * side_loss + .1 * refine_loss + .1 * viability_loss
        return MiniTrainingOutput(total, core_loss.total, side_loss, refine_loss, viability_loss)
