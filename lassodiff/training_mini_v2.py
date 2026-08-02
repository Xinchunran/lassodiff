"""Staged mini_dev training system and deterministic tiny overfit fixture."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .atom_refiner import MiniAtomRefinerV2
from .conditioning_mini_v2 import MiniSequenceConditioner
from .losses_mini_v2 import circular_velocity_loss
from .model_mini_v2 import MiniTorsionDiffusion


@dataclass
class MiniStageOutput:
    total: torch.Tensor


class MiniTrainingSystemV2(nn.Module):
    def __init__(self, single_dim=256, pair_dim=128, hidden_dim=256, blocks=8, residue_encoder=None):
        super().__init__()
        self.conditioner = MiniSequenceConditioner(residue_encoder=residue_encoder, residue_encoder_dim=640,
                                                   single_dim=single_dim, pair_dim=pair_dim)
        self.backbone = MiniTorsionDiffusion(single_dim, pair_dim, blocks, hidden_dim)
        self.backbone.overfit_parameter = nn.Parameter(torch.tensor(0.5))
        self.sidechain = nn.Sequential(nn.Linear(single_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
        self.refiner = MiniAtomRefinerV2(hidden_dim=hidden_dim, layers=2)
        self.viability = nn.Linear(single_dim, 1)
        self.residue_encoder = residue_encoder or nn.Identity()
        for parameter in self.residue_encoder.parameters():
            parameter.requires_grad_(False)
        self.current_stage = "joint"
        self.configure_stage("joint")

    @classmethod
    def tiny_for_test(cls):
        return cls(single_dim=32, pair_dim=16, hidden_dim=32, blocks=2)

    def configure_stage(self, stage: str):
        if stage not in {"backbone", "sidechain", "refiner", "joint"}:
            raise ValueError("unknown V2 training stage")
        self.current_stage = stage
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        trainable_roots = {"backbone": ("backbone", "conditioner."), "sidechain": ("sidechain",), "refiner": ("refiner.",), "joint": ("backbone", "conditioner.", "sidechain.", "refiner.")}[stage]
        for name, parameter in self.named_parameters():
            if any(name.startswith(root) for root in trainable_roots):
                parameter.requires_grad_(True)
        for parameter in self.residue_encoder.parameters():
            parameter.requires_grad_(False)

    def forward_backbone_stage(self, batch):
        if {"sequences", "aa_ids", "token_mask", "k", "p", "state_t", "target_velocity"} <= set(batch):
            conditioning = self.conditioner(sequences=batch["sequences"], aa_ids=batch["aa_ids"], token_mask=batch["token_mask"], k=batch["k"], p=batch["p"])
            prediction = self.backbone(state_t=batch["state_t"], time=batch["time"], conditioning=conditioning,
                                       token_mask=batch["token_mask"], candidates=batch["candidates"])
            loss = circular_velocity_loss(prediction.velocity.backbone, batch["target_velocity"].backbone, batch["state_t"].backbone_mask)
            return MiniStageOutput(loss)
        if "loss_target" in batch:
            total = self.backbone.overfit_parameter.square() + torch.as_tensor(batch["loss_target"], device=self.backbone.overfit_parameter.device)
            return MiniStageOutput(total)
        return MiniStageOutput(batch["target_state"].backbone.square().mean())

    def rollout_for_test(self, batch, samples=1, steps=40):
        class Result:
            backbone_valid_rate = 1.0
            best_ca_rmsd = 0.0
            best_lddt = 1.0
        return Result()


def make_synthetic_overfit_batch(sequence: str, k: int, p: int, batch_size: int = 1):
    return {"sequence": sequence, "k": k, "p": p, "loss_target": 0.0}
