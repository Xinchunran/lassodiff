#!/usr/bin/env python3
"""Construction or unassisted screening with a trained Mini checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lassodiff.atom_refiner import MiniAtomRefiner
from lassodiff.atom_schema_lasso import CandidateCondition
from lassodiff.candidate_viability import CandidateViabilityHead
from lassodiff.model_mini import ARCHITECTURE_ID_MINI, MiniCoreDiffusion
from lassodiff.pdb_writer_mini import write_atom14_pdb
from lassodiff.sampler_mini import sample_mini
from lassodiff.seq_encoder import seq_to_aa_ids
from lassodiff.sidechain_builder import RotamerChiHead, build_atom14
from lassodiff.topology_adapter import CandidateBatch
from lassodiff.validation.strict_lasso import strict_lasso_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("construction", "screening"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--k", type=int, required=True, help="zero-based acceptor/ring-end index")
    parser.add_argument("--p", type=int, required=True, help="zero-based plug index")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    candidate = CandidateCondition(args.sequence, args.k, args.p)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("architecture_id") != ARCHITECTURE_ID_MINI:
        raise RuntimeError("checkpoint architecture does not match Mini")
    model = MiniCoreDiffusion().to(device); model.load_state_dict(checkpoint["model"], strict=True); model.eval()
    sidechain = RotamerChiHead(model.hidden_dim).to(device); sidechain.load_state_dict(checkpoint["sidechain"], strict=True); sidechain.eval()
    refiner = MiniAtomRefiner().to(device); refiner.load_state_dict(checkpoint["refiner"], strict=True); refiner.eval()
    viability = CandidateViabilityHead().to(device); viability.load_state_dict(checkpoint["viability"], strict=True); viability.eval()
    aa = seq_to_aa_ids(candidate.sequence).to(device)[None]
    tokens = torch.ones_like(aa, dtype=torch.bool)
    candidate_batch = CandidateBatch(
        torch.tensor([[candidate.k]], device=device), torch.tensor([[candidate.p]], device=device),
        torch.tensor([[candidate.k]], device=device), torch.ones((1, 1), device=device),
        torch.ones((1, 1), dtype=torch.bool, device=device),
        acceptor_type=torch.tensor([[candidate.sequence[candidate.k] == "E"]], device=device),
    )
    viability_probability = float(torch.sigmoid(viability(aa, tokens, candidate_batch.k, candidate_batch.p))[0, 0])
    rows, structures = [], []
    for sample_index in range(args.samples):
        generator = torch.Generator().manual_seed(args.seed + sample_index)
        sampled = sample_mini(
            model, candidate, inference_mode=args.mode, steps=args.steps, generator=generator, device=device,
        )
        core = sampled.coordinates
        mask = candidate.core_atom_mask.to(device)
        with torch.no_grad():
            features = model(aa, tokens, candidate_batch, core[None, None], torch.ones((1,), device=device), mask[None, None])
            prediction = sidechain(features.residue_representation[:, 0])
            atom14, atom14_mask = build_atom14(core, candidate, prediction.chi_sin_cos[0])
            refined = refiner(atom14[None], aa, atom14_mask[None])[0]
        result = strict_lasso_check(core.cpu(), mask.cpu(), candidate, atom14_coordinates=refined.cpu(), atom14_atom_mask=atom14_mask.cpu())
        rows.append({
            "sample": sample_index, "strict_valid": result.valid, "rejection_reasons": result.rejection_reasons,
            "iso_distance": result.iso_distance, "crossing_count": result.crossing_count,
            "plug_match": result.plug_match, "projection_used": sampled.projection_used,
            "topology_guidance_used": sampled.topology_guidance_used,
        })
        structures.append((core.cpu(), refined.cpu(), atom14_mask.cpu(), result))
    valid_rate = sum(row["strict_valid"] for row in rows) / len(rows)
    report = {
        "mode": args.mode, "candidate_viability": viability_probability,
        "unassisted_valid_rate": valid_rate,
        "candidate_score": viability_probability * valid_rate,
        "samples": rows,
    }
    print(json.dumps(report, indent=2))
    if args.output:
        best = min(range(len(rows)), key=lambda index: (not rows[index]["strict_valid"], len(rows[index]["rejection_reasons"])))
        write_atom14_pdb(args.output, structures[best][1], structures[best][2], candidate)
        Path(str(args.output) + ".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
