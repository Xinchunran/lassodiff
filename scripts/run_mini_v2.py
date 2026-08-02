#!/usr/bin/env python3
"""Generate and score an explicit unassisted or assisted V2 rollout."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.checkpoint_mini_v2 import load_mini_v2_checkpoint
from lassodiff.data.mini_grouped_dataset import collate_grouped_mini
from lassodiff.evaluation_mini_v2 import evaluate_generated_candidate
from lassodiff.sampler_mini_v2 import MiniInferenceConfig, sample_torsion_model
from lassodiff.training_mini_v2 import MiniTrainingSystemV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--sequence", required=True)
    parser.add_argument("--k", type=int, required=True); parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--samples", type=int, default=8); parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--mode", choices=("unassisted", "assisted"), default="unassisted"); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate = CandidateCondition(args.sequence, args.k, args.p)
    system = MiniTrainingSystemV2(); load_mini_v2_checkpoint(args.checkpoint, system=system)
    system.eval(); device = next(system.parameters()).device
    aa = torch.tensor([["ACDEFGHIKLMNPQRSTVWY".index(a) for a in candidate.sequence]], dtype=torch.long, device=device)
    mask = torch.ones((1, len(candidate.sequence)), dtype=torch.bool, device=device)
    config = MiniInferenceConfig.unassisted(args.steps) if args.mode == "unassisted" else MiniInferenceConfig.assisted(args.steps)
    output = sample_torsion_model(system.backbone, system.conditioner, [candidate], aa, mask,
                                  config=config, steps=args.steps, num_samples=args.samples,
                                  generator=torch.Generator().manual_seed(17), device=device)
    rows = []
    for sample, finite in zip(output.core_coordinates[0], output.finite[0]):
        sample_mask = candidate.core_atom_mask.to(sample.device)
        rows.append(evaluate_generated_candidate(sample, sample_mask, candidate))
    report = {"architecture_id": system.architecture_id, "schema_version": 2, "prior_mode": config.prior_mode,
              "projection_used": config.projection, "topology_seeded": config.topology_seeded,
              "samples": rows, "finite_rate": float(output.finite.float().mean())}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__": main()
