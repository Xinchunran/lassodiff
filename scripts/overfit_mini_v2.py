#!/usr/bin/env python3
"""Run the deterministic synthetic overfit gate."""
from __future__ import annotations

import argparse

import torch

from lassodiff.training_mini_v2 import MiniTrainingSystemV2, make_synthetic_overfit_batch


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=500); args = parser.parse_args()
    system = MiniTrainingSystemV2.tiny_for_test(); system.configure_stage("backbone")
    batch = make_synthetic_overfit_batch("AAADRAAA", 3, 5)
    optimizer = torch.optim.AdamW([p for p in system.parameters() if p.requires_grad], lr=2e-3)
    initial = None
    for _ in range(args.steps):
        loss = system.forward_backbone_stage(batch).total
        initial = float(loss.detach()) if initial is None else initial
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    print({"initial": initial, "final": float(loss.detach())})


if __name__ == "__main__":
    main()
