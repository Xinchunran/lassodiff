#!/usr/bin/env python3
"""Behavioral architecture preflight required before Mini training."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch

from lassodiff.atom_refiner import MiniAtomRefiner
from lassodiff.atom_schema_lasso import ATOM_CA, CandidateCondition, MINI_SCHEMA_VERSION
from lassodiff.candidate_viability import CandidateViabilityHead
from lassodiff.model_mini import ARCHITECTURE_ID_MINI, MiniCoreDiffusion
from lassodiff.peptide_prior import sample_peptide_prior
from lassodiff.sampler_mini import sample_mini
from lassodiff.seq_encoder import seq_to_aa_ids
from lassodiff.sidechain_builder import build_atom14
from lassodiff.structure_processor import isopeptide_distance, process_lasso_structure
from lassodiff.topology_adapter import CandidateBatch
from lassodiff.training_mini import MiniTrainingOutput, MiniTrainingSystem
from lassodiff.validation.threading_mini import hard_threading_check_ca


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--min1", default="structure/LP_WP_069782233/min1.pdb")
    args = parser.parse_args()
    torch.manual_seed(7)
    candidate = CandidateCondition("LLQRNGRDRLILSKN", 7, 9)
    generator = torch.Generator().manual_seed(11)
    priors = {
        "open": sample_peptide_prior(candidate, mode="open_chain", generator=generator, noise_scale=0),
        "single": sample_peptide_prior(candidate, mode="single_crossing", generator=generator, noise_scale=0),
        "zero": sample_peptide_prior(candidate, mode="topology_corrupted", corruption="no_crossing", generator=generator, noise_scale=0),
        "double": sample_peptide_prior(candidate, mode="topology_corrupted", corruption="double_crossing", generator=generator, noise_scale=0),
    }
    batch_candidate = CandidateBatch(
        torch.tensor([[candidate.k]]), torch.tensor([[candidate.p]]), torch.tensor([[candidate.k]]),
        torch.ones((1, 1)), torch.ones((1, 1), dtype=torch.bool), acceptor_type=torch.zeros((1, 1), dtype=torch.long),
    )
    counts = {}
    for name, prior in priors.items():
        check = hard_threading_check_ca(
            prior.coordinates[:, ATOM_CA][None, None], torch.ones((1, len(candidate.sequence)), dtype=torch.bool),
            batch_candidate, prior.atom_mask[:, ATOM_CA][None, None],
        )
        counts[name] = int(check.crossing_count[0, 0])
    if counts["single"] != 1 or counts["zero"] != 0 or counts["double"] < 2:
        raise RuntimeError(f"programmatic prior topology preflight failed: {counts}")

    model = MiniCoreDiffusion(hidden_dim=32, blocks=2)
    aa = seq_to_aa_ids(candidate.sequence)[None]
    tokens = torch.ones_like(aa, dtype=torch.bool)
    x = priors["single"].coordinates[None, None]
    atom_mask = priors["single"].atom_mask[None, None]
    output = model(aa, tokens, batch_candidate, x, torch.tensor([.5]), atom_mask)
    shifted = model(aa, tokens, batch_candidate, x + .2 * torch.randn_like(x), torch.tensor([.5]), atom_mask)
    if torch.allclose(output.velocity, shifted.velocity):
        raise RuntimeError("Mini diffusion output does not depend on current coordinates")
    sampled = sample_mini(model, candidate, inference_mode="screening", steps=3, generator=torch.Generator().manual_seed(2))
    if sampled.prior_mode != "open_chain" or sampled.projection_used or sampled.topology_guidance_used:
        raise RuntimeError("screening integrity preflight failed")
    if "coord" in " ".join(inspect.signature(CandidateViabilityHead.forward).parameters).lower():
        raise RuntimeError("viability head must not accept generated coordinates")
    training_system = MiniTrainingSystem(hidden_dim=32, diffusion_blocks=2)
    state_keys = tuple(training_system.state_dict())
    required_prefixes = ("model.", "sidechain.", "refiner.", "viability.")
    if not all(any(key.startswith(prefix) for key in state_keys) for prefix in required_prefixes):
        raise RuntimeError("unified FSDP root is missing a required trainable module")
    if getattr(MiniTrainingOutput.__dataclass_params__, "frozen", True):
        raise RuntimeError("FSDP output must permit recursive backward-hook registration")

    atom14, atom14_mask = build_atom14(priors["single"].coordinates, candidate)
    refiner = MiniAtomRefiner(hidden_dim=24, layers=4, max_displacement=.5)
    refined = refiner(atom14[None], aa, atom14_mask[None])
    displacement = (refined - atom14[None]).norm(dim=-1)[atom14_mask[None]]
    if not bool(torch.isfinite(refined).all()) or float(displacement.max()) > .5001:
        raise RuntimeError("all-heavy refiner displacement contract failed")

    chemistry = None
    min1 = Path(args.min1)
    if min1.is_file():
        structure = process_lasso_structure(min1, candidate)
        chemistry = {
            "normalized_acceptor": structure.residue_names[candidate.k],
            "iso_distance_angstrom": isopeptide_distance(structure, candidate),
            "single_carbonyl_oxygen": bool(structure.core_atom_mask[candidate.k, 6]),
        }
    report = {
        "status": "PASS", "architecture_id": ARCHITECTURE_ID_MINI,
        "schema_version": MINI_SCHEMA_VERSION, "template_coordinates_used": False,
        "prior_crossing_counts": counts, "screening_projection": False,
        "screening_topology_guidance": False, "dynamic_geometry_calls": output.dynamic_geometry_calls,
        "distributed_backend": "fsdp_full_shard",
        "fsdp_root_modules": [prefix[:-1] for prefix in required_prefixes],
        "fsdp_full_state_checkpoint": True,
        "chemistry_fixture": chemistry,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
