#!/usr/bin/env python3
"""FSDP-capable real Mini V2 trainer; no legacy Mini fallback."""
from __future__ import annotations
import argparse, json, os, random, subprocess
from pathlib import Path
import torch
from torch.utils.data import DataLoader, DistributedSampler

from lassodiff.checkpoint_mini_v2 import load_mini_v2_checkpoint, save_mini_v2_checkpoint, validate_resume_checkpoint
from lassodiff.data.mini_grouped_dataset import collate_grouped_mini
from lassodiff.data.mini_grouped_pdb_dataset import GroupedMiniPDBDataset
from lassodiff.training_mini_v2 import MiniTrainingSystemV2
from lassodiff.esm_encoder_mini import CachedESMResidueEncoder
from scripts.verify_mini_v2_startup import _require_v2_preflight


def _dist():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = torch.distributed.get_rank(); world = torch.distributed.get_world_size()
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1)) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda": torch.cuda.set_device(device)
    return rank, world, device


def _fold(split, fold):
    for item in split["folds"]:
        if int(item["fold"]) == fold: return item
    raise ValueError(f"unknown fold {fold}")


def _source_commit():
    try:
        value = subprocess.check_output(["git", "--git-dir=.git-mini-dev", "rev-parse", "HEAD"], text=True).strip()
        if len(value) == 40:
            return value
    except (OSError, subprocess.CalledProcessError):
        pass
    raise RuntimeError("cannot record the V2 source commit")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True); parser.add_argument("--structure-root", required=True)
    parser.add_argument("--esm-cache", default=None)
    parser.add_argument("--preflight", required=True); parser.add_argument("--split", required=True)
    parser.add_argument("--source-split", required=True); parser.add_argument("--config", default="configs/lassodiff_mini_v2.yaml")
    parser.add_argument("--fold", type=int, required=True); parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", choices=("backbone", "sidechain", "refiner", "joint"), default="backbone")
    parser.add_argument("--resume", default=None); parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17); parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500); parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--rollout-every", type=int, default=1000); parser.add_argument("--validation-max-batches", type=int, default=0)
    parser.add_argument("--rollout-candidates", type=int, default=16); parser.add_argument("--rollout-samples", type=int, default=4); parser.add_argument("--rollout-steps", type=int, default=40)
    args = parser.parse_args()
    rank, world, device = _dist()
    cv, source = json.loads(Path(args.split).read_text()), json.loads(Path(args.source_split).read_text())
    preflight = _require_v2_preflight(args.preflight, cv, source)
    selected = _fold(cv, args.fold)
    train = GroupedMiniPDBDataset(args.metadata, args.structure_root, record_ids=selected["train"])
    valid = GroupedMiniPDBDataset(args.metadata, args.structure_root, record_ids=selected["val"])
    if Path(args.run_dir).exists() and any(Path(args.run_dir).iterdir()) and args.resume is None:
        raise RuntimeError("refusing to write into a non-empty V2 run directory")
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed + rank); random.seed(args.seed + rank)
    encoder = CachedESMResidueEncoder(args.esm_cache) if args.esm_cache else None
    system = MiniTrainingSystemV2(residue_encoder=encoder).to(device); system.configure_stage(args.stage)
    if world > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        system = FSDP(system, device_id=device, use_orig_params=True, sync_module_states=True, limit_all_gathers=True)
    sampler = DistributedSampler(train, shuffle=True) if world > 1 else None
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.num_workers, collate_fn=collate_grouped_mini)
    parameters = [p for p in system.parameters() if p.requires_grad]
    if not parameters: raise RuntimeError(f"stage {args.stage} has no trainable parameters")
    lr = {"backbone": 2e-4, "sidechain": 1e-4, "refiner": 1e-4, "joint": 2e-5}[args.stage]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-2)
    start = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_checkpoint(checkpoint, stage=args.stage, cv_split_manifest_sha256=cv["manifest_sha256"],
                                   source_split_manifest_sha256=source["manifest_sha256"], dataset_mapping_sha256=train.mapping_sha256)
        load_mini_v2_checkpoint(args.resume, system=system)
        optimizer.load_state_dict(checkpoint["optimizer"]); start = int(checkpoint["step"])
    steps = args.steps or {"backbone": 10000, "sidechain": 4000, "refiner": 3000, "joint": 3000}[args.stage]
    generator = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    log_path = Path(args.run_dir) / "metrics.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        iterator = iter(loader)
        for step in range(start + 1, steps + 1):
            if sampler is not None and step == start + 1: sampler.set_epoch(step)
            try: batch = next(iterator)
            except StopIteration: iterator = iter(loader); batch = next(iterator)
            output = system.module.forward_backbone_stage(batch, generator=generator, global_step=step) if world > 1 else system.forward_backbone_stage(batch, generator=generator, global_step=step)
            if not torch.isfinite(output.total): raise RuntimeError(f"non-finite V2 loss at step {step}")
            optimizer.zero_grad(set_to_none=True); output.total.backward()
            grad_norm = system.clip_grad_norm_(1.0) if world > 1 else torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not torch.isfinite(torch.as_tensor(grad_norm, device=device)): raise RuntimeError("non-finite V2 gradient")
            optimizer.step()
            row = {"architecture_id": "lassodiff_mini_torsion_v2", "schema_version": 2, "stage": args.stage,
                   "fold": args.fold, "step": step, "world_size": world, "global_sample_count": step * args.batch_size * world,
                   "total": float(output.total.detach()), "backbone_flow": float(output.backbone_flow.detach()),
                   "acceptor_chi_flow": float(output.acceptor_chi_flow.detach()), "endpoint_core": float(output.endpoint_core.detach()),
                   "endpoint_pair": float(output.endpoint_pair.detach()), "iso_geometry": float(output.iso_geometry.detach()),
                   "topology": float(output.topology.detach()), "clash": float(output.clash.detach()),
                   "sidechain": float(output.sidechain.detach()), "refine": float(output.refine.detach()),
                   "viability": float(output.viability.detach()), "grad_norm": float(grad_norm), "learning_rate": lr}
            if rank == 0 and (step % args.log_every == 0 or step == 1): log.write(json.dumps(row) + "\n"); log.flush()
            if step % args.save_every == 0 or step == steps:
                if rank == 0:
                    provenance = {"source_commit": _source_commit(), "encoder_name": "esm2_t30_150M_UR50D", "encoder_revision": "main", "encoder_frozen": True,
                                  "grouped_target_mapping_sha256": train.mapping_sha256, "dataset_mapping_sha256": train.mapping_sha256,
                                  "validation_mapping_sha256": valid.mapping_sha256, "cv_split_manifest_sha256": cv["manifest_sha256"], "source_split_manifest_sha256": source["manifest_sha256"]}
                    save_mini_v2_checkpoint(Path(args.run_dir) / ("checkpoint-final.pt" if step == steps else f"checkpoint-{step}.pt"),
                                            system=system, optimizer=optimizer, step=step, stage=args.stage, world_size=world, preflight=preflight, provenance=provenance)
    if world > 1: torch.distributed.barrier(); torch.distributed.destroy_process_group()


if __name__ == "__main__": main()
