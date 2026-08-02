#!/usr/bin/env python3
"""Three-stage Mini trainer; requires a passing behavioral preflight report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lassodiff.atom_refiner import MiniAtomRefiner
from lassodiff.candidate_viability import CandidateViabilityHead
from lassodiff.data.mini_dataset import MiniLassoDataset, collate_mini
from lassodiff.flow_matching import align_target_to_source, flow_interpolate
from lassodiff.losses_mini import mini_core_loss
from lassodiff.model_mini import ARCHITECTURE_ID_MINI, MiniCoreDiffusion
from lassodiff.peptide_prior import sample_peptide_prior, sample_prior_mode
from lassodiff.sidechain_builder import RotamerChiHead, build_atom14, symmetry_aware_coordinate_loss
from lassodiff.topology_adapter import CandidateBatch


def _require_preflight(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("architecture_id") != ARCHITECTURE_ID_MINI:
        raise RuntimeError("Mini training requires a matching PASS preflight")
    if report.get("template_coordinates_used") is not False or report.get("screening_projection") is not False:
        raise RuntimeError("Mini preflight violates template/screening contract")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--structure-root", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    report = _require_preflight(Path(args.preflight))
    dataset = MiniLassoDataset(args.metadata, args.structure_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mini)
    device = torch.device(args.device)
    model = MiniCoreDiffusion().to(device)
    sidechain = RotamerChiHead(model.hidden_dim).to(device)
    refiner = MiniAtomRefiner().to(device)
    viability = CandidateViabilityHead().to(device)
    trainable = list(model.parameters()) + list(sidechain.parameters()) + list(refiner.parameters()) + list(viability.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)
    generator = torch.Generator().manual_seed(args.seed)
    iterator = iter(loader)
    metrics = []
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        B, _M, L, _A, _ = batch["core"].shape
        x0 = torch.zeros_like(batch["core"])
        for b, condition in enumerate(batch["conditions"]):
            mode = sample_prior_mode(generator)
            prior = sample_peptide_prior(condition, mode=mode, generator=generator)
            x0[b, 0, :len(condition.sequence)] = prior.coordinates
        aa_ids = batch["aa_ids"].to(device)
        token_mask = batch["token_mask"].to(device)
        target = batch["core"].to(device)
        core_mask = batch["core_mask"].to(device)
        x0 = x0.to(device)
        candidates = CandidateBatch(
            batch["k"].to(device), batch["p"].to(device), batch["k"].to(device),
            torch.ones((B, 1), device=device), torch.ones((B, 1), dtype=torch.bool, device=device),
            acceptor_type=batch["acceptor_type"].to(device),
        )
        t = torch.rand((B,), generator=generator).to(device).clamp(.02, .98)
        x_t, target_velocity, aligned_core = flow_interpolate(x0, target, t, core_mask)
        output = model(aa_ids, token_mask, candidates, x_t, t, core_mask)
        core_loss = mini_core_loss(
            output.velocity, target_velocity, x_t, t, aligned_core, token_mask, core_mask, candidates,
        )
        endpoint = x_t + (1 - t[:, None, None, None, None]) * output.velocity
        side_prediction = sidechain(output.residue_representation[:, 0])
        predicted_atom14, predicted_masks = [], []
        for b, condition in enumerate(batch["conditions"]):
            built, mask = build_atom14(
                endpoint[b, 0, :len(condition.sequence)], condition,
                side_prediction.chi_sin_cos[b, :len(condition.sequence)],
            )
            padded = endpoint.new_zeros((L, 14, 3)); padded[:built.shape[0]] = built
            padded_mask = torch.zeros((L, 14), dtype=torch.bool, device=device); padded_mask[:mask.shape[0]] = mask
            predicted_atom14.append(padded); predicted_masks.append(padded_mask)
        predicted_atom14 = torch.stack(predicted_atom14)
        predicted_masks = torch.stack(predicted_masks)
        target14 = batch["atom14"].to(device)
        target14_mask = batch["atom14_mask"].to(device) & predicted_masks
        aligned_target14 = align_target_to_source(predicted_atom14, target14, target14_mask)
        side_loss = endpoint.sum() * 0
        for b, condition in enumerate(batch["conditions"]):
            side_loss = side_loss + symmetry_aware_coordinate_loss(
                predicted_atom14[b, :len(condition.sequence)], aligned_target14[b, :len(condition.sequence)],
                condition.sequence, target14_mask[b, :len(condition.sequence)], condition,
            )
        side_loss = side_loss / B
        refined = refiner(predicted_atom14, aa_ids, predicted_masks)
        refine_loss = ((refined - aligned_target14).square().sum(-1) * target14_mask).sum() / target14_mask.sum().clamp_min(1)
        correct_p = batch["p"].to(device)
        negative_p = torch.where(correct_p + 1 < token_mask.sum(1, keepdim=True) - 1, correct_p + 1, correct_p - 1)
        correct_logit = viability(aa_ids, token_mask, batch["k"].to(device), correct_p)
        negative_logit = viability(aa_ids, token_mask, batch["k"].to(device), negative_p)
        viability_loss = torch.nn.functional.binary_cross_entropy_with_logits(correct_logit, torch.ones_like(correct_logit))
        finite_negative = torch.isfinite(negative_logit)
        if bool(finite_negative.any()):
            viability_loss = viability_loss + torch.nn.functional.binary_cross_entropy_with_logits(
                negative_logit[finite_negative], torch.zeros_like(negative_logit[finite_negative]),
            )
        total = core_loss.total + .2 * side_loss + .1 * refine_loss + .1 * viability_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % 10 == 0:
            row = {"step": step, "total": float(total), "core": float(core_loss.total), "sidechain": float(side_loss), "refine": float(refine_loss), "viability": float(viability_loss)}
            if not all(torch.isfinite(torch.tensor(value)) for key, value in row.items() if key != "step"):
                raise RuntimeError("non-finite Mini training metric")
            metrics.append(row); print(json.dumps(row, sort_keys=True))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture_id": ARCHITECTURE_ID_MINI, "model": model.state_dict(),
        "sidechain": sidechain.state_dict(), "refiner": refiner.state_dict(),
        "viability": viability.state_dict(),
        "optimizer": optimizer.state_dict(), "step": args.steps, "preflight": report,
    }, run_dir / "checkpoint-final.pt")
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics), encoding="utf-8")


if __name__ == "__main__":
    main()
