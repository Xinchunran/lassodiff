"""Canonical torsion diffusion state used by the mini_dev generator."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TorsionState:
    backbone: torch.Tensor
    backbone_mask: torch.Tensor
    acceptor_chi: torch.Tensor
    acceptor_chi_mask: torch.Tensor

    def to(self, device=None, dtype=None) -> "TorsionState":
        def move(value):
            return value.to(device=device, dtype=dtype if value.is_floating_point() else None)

        return TorsionState(
            move(self.backbone), move(self.backbone_mask),
            move(self.acceptor_chi), move(self.acceptor_chi_mask),
        )

    def clone(self) -> "TorsionState":
        return TorsionState(
            self.backbone.clone(), self.backbone_mask.clone(),
            self.acceptor_chi.clone(), self.acceptor_chi_mask.clone(),
        )


@dataclass
class TorsionVelocity:
    backbone: torch.Tensor
    acceptor_chi: torch.Tensor

    def to(self, device=None, dtype=None) -> "TorsionVelocity":
        def move(value):
            return value.to(device=device, dtype=dtype if value.is_floating_point() else None)

        return TorsionVelocity(move(self.backbone), move(self.acceptor_chi))


def validate_torsion_state(state: TorsionState) -> None:
    if state.backbone.shape[-1] != 3 or state.acceptor_chi.shape[-1] != 4:
        raise ValueError("torsion state must contain [phi,psi,omega] and four chi slots")
    if state.backbone_mask.shape != state.backbone.shape:
        raise ValueError("backbone mask shape mismatch")
    if state.acceptor_chi_mask.shape != state.acceptor_chi.shape:
        raise ValueError("acceptor chi mask shape mismatch")
