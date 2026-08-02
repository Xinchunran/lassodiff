#!/usr/bin/env python3
"""Three-stage Mini trainer; requires a passing behavioral preflight report."""
from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)
from torch.utils.data import DataLoader, DistributedSampler

from lassodiff.data.mini_dataset import MiniLassoDataset, collate_mini
from lassodiff.data.mini_split import select_mini_cv_fold, validate_mini_cv_manifest
from lassodiff.model_mini import ARCHITECTURE_ID_MINI
from lassodiff.peptide_prior import sample_peptide_prior, sample_prior_mode
from lassodiff.training_mini import MiniTrainingSystem


def _require_preflight(path: Path, cv_split: dict, source_split: dict):
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("architecture_id") != ARCHITECTURE_ID_MINI:
        raise RuntimeError("Mini training requires a matching PASS preflight")
    if report.get("template_coordinates_used") is not False or report.get("screening_projection") is not False:
        raise RuntimeError("Mini preflight violates template/screening contract")
    if report.get("distributed_backend") != "fsdp_full_shard" or report.get("fsdp_full_state_checkpoint") is not True:
        raise RuntimeError("Mini preflight does not authorize FSDP full-state training")
    expected_split = (
        cv_split.get("manifest_sha256"), source_split.get("manifest_sha256"),
        cv_split.get("fold_count"), cv_split.get("locked_test_record_count"),
    )
    reported_split = (
        report.get("cv_split_manifest_sha256"), report.get("source_split_manifest_sha256"),
        report.get("cv_fold_count"), report.get("locked_test_record_count"),
    )
    if reported_split != expected_split:
        raise RuntimeError("Mini preflight does not match the locked cross-validation split")
    return report


def _distributed_device(requested: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if requested == "cpu" or not torch.cuda.is_available():
            raise RuntimeError("distributed Mini training requires CUDA/NCCL")
        if not 0 <= local_rank < torch.cuda.device_count():
            raise RuntimeError("LOCAL_RANK is outside visible CUDA devices")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return torch.device("cuda", local_rank), rank, world_size, local_rank
    device = torch.device(requested)
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    return device, rank, world_size, local_rank


def _reduce_metrics(values, batch_size, device, world_size):
    packed = torch.tensor([*(float(value.detach()) * batch_size for value in values), batch_size], device=device)
    if world_size > 1:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return [float(value / packed[-1].clamp_min(1)) for value in packed[:-1]], int(packed[-1])


def _save_checkpoint(path, *, system, optimizer, step, report, world_size, rank, provenance):
    if isinstance(system, FSDP):
        state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optimizer_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            system, StateDictType.FULL_STATE_DICT, state_config, optimizer_config,
        ):
            model_state = system.state_dict()
            optimizer_state = FSDP.optim_state_dict(system, optimizer)
    else:
        model_state = system.state_dict()
        optimizer_state = optimizer.state_dict()
    if rank == 0:
        torch.save({
            "architecture_id": ARCHITECTURE_ID_MINI, "system": model_state,
            "optimizer": optimizer_state, "step": int(step),
            "world_size": int(world_size), "distributed_backend": "fsdp_full_shard" if world_size > 1 else "none",
            "preflight": report, "provenance": provenance,
        }, path)


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _forward_batch(system, batch, *, device, generator, rank, step, phase):
    B = batch["core"].shape[0]
    x0 = torch.zeros_like(batch["core"])
    for b, condition in enumerate(batch["conditions"]):
        mode = sample_prior_mode(generator)
        try:
            prior = sample_peptide_prior(condition, mode=mode, generator=generator)
        except Exception as exc:
            raise RuntimeError(
                "procedural prior failed for "
                f"phase={phase}, rank={rank}, step={step}, batch_index={b}, mode={mode}, "
                f"sequence={condition.sequence}, k={condition.k}, p={condition.p}"
            ) from exc
        x0[b, 0, :len(condition.sequence)] = prior.coordinates
    aa_ids = batch["aa_ids"].to(device, non_blocking=True)
    token_mask = batch["token_mask"].to(device, non_blocking=True)
    target = batch["core"].to(device, non_blocking=True)
    core_mask = batch["core_mask"].to(device, non_blocking=True)
    t = torch.rand((B,), generator=generator).to(device).clamp(.02, .98)
    output = system(
        aa_ids=aa_ids, token_mask=token_mask, target=target, core_mask=core_mask,
        target14=batch["atom14"].to(device, non_blocking=True),
        target14_mask=batch["atom14_mask"].to(device, non_blocking=True),
        k=batch["k"].to(device, non_blocking=True), p=batch["p"].to(device, non_blocking=True),
        acceptor_type=batch["acceptor_type"].to(device, non_blocking=True),
        conditions=batch["conditions"], x0=x0.to(device, non_blocking=True), t=t,
    )
    return output, B


def _validate(system, loader, *, device, rank, world_size, seed, fold, step, max_batches):
    generator = torch.Generator().manual_seed(seed + 700001 + fold * 1009 + rank * 100003)
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    system.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            output, batch_size = _forward_batch(
                system, batch, device=device, generator=generator, rank=rank,
                step=step, phase="validation",
            )
            values = (output.total, output.core, output.sidechain, output.refine, output.viability)
            if not all(bool(torch.isfinite(value.detach())) for value in values):
                raise RuntimeError("non-finite Mini validation loss")
            totals[:5] += torch.stack([value.detach().double() for value in values]) * batch_size
            totals[5] += batch_size
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    system.train()
    if totals[5] <= 0:
        raise RuntimeError("Mini validation evaluated zero examples")
    return [float(value / totals[5]) for value in totals[:5]], int(totals[5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--structure-root", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-max-batches", type=int, default=0)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.log_every, args.save_every, args.validate_every) < 1:
        raise ValueError("steps, batch-size, log/save/validate intervals must be positive")
    if args.validation_max_batches < 0:
        raise ValueError("validation-max-batches must be non-negative")
    source_split = json.loads(Path(args.source_split).read_text(encoding="utf-8"))
    cv_split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    validate_mini_cv_manifest(cv_split, source_split)
    selected_fold = select_mini_cv_fold(cv_split, args.fold)
    report = _require_preflight(Path(args.preflight), cv_split, source_split)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True,
    ).strip()
    if len(source_commit) != 40:
        raise RuntimeError("cannot resolve a full source commit for Mini training")
    preflight_sha256 = hashlib.sha256(Path(args.preflight).read_bytes()).hexdigest()
    device, rank, world_size, local_rank = _distributed_device(args.device)
    torch.manual_seed(args.seed + rank * 100003)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank * 100003)
    dataset = MiniLassoDataset(args.metadata, args.structure_root, record_ids=selected_fold["train"])
    validation_dataset = MiniLassoDataset(
        args.metadata, args.structure_root, record_ids=selected_fold["val"],
    )
    unavailable = set(cv_split["unavailable_record_ids"])
    expected_train_missing = unavailable & set(selected_fold["train"])
    expected_validation_missing = unavailable & set(selected_fold["val"])
    if set(dataset.missing_record_ids) != expected_train_missing:
        raise RuntimeError("Mini train availability differs from the locked CV manifest")
    if set(validation_dataset.missing_record_ids) != expected_validation_missing:
        raise RuntimeError("Mini validation availability differs from the locked CV manifest")
    if set(dataset.qualified_record_ids) & set(validation_dataset.qualified_record_ids):
        raise RuntimeError("Mini train and validation qualified records overlap")
    if world_size > 1:
        identities = [None for _ in range(world_size)]
        dist.all_gather_object(
            identities,
            (
                len(dataset), len(dataset.rejections), dataset.mapping_sha256,
                len(validation_dataset), len(validation_dataset.rejections), validation_dataset.mapping_sha256,
                dataset.missing_record_ids, validation_dataset.missing_record_ids,
            ),
        )
        if len(set(identities)) != 1:
            raise RuntimeError(f"Mini dataset identity differs across ranks: {identities}")
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed,
    ) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
        collate_fn=collate_mini, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    validation_sampler = DistributedSampler(
        validation_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True,
    ) if world_size > 1 else None
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size,
        shuffle=False, sampler=validation_sampler, collate_fn=collate_mini,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    system = MiniTrainingSystem().to(device)
    if world_size > 1:
        system = FSDP(
            system, device_id=device, use_orig_params=True,
            sync_module_states=True, limit_all_gathers=True,
        )
    trainable = list(system.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)
    generator = torch.Generator().manual_seed(args.seed + rank * 100003)
    run_dir = Path(args.run_dir)
    run_state = {"completed": False, "step": 0}
    provenance = {
        "source_commit": source_commit,
        "preflight_sha256": preflight_sha256,
        "dataset_mapping_sha256": dataset.mapping_sha256,
        "validation_mapping_sha256": validation_dataset.mapping_sha256,
        "cv_split_manifest_sha256": cv_split["manifest_sha256"],
        "source_split_manifest_sha256": source_split["manifest_sha256"],
        "fold": args.fold,
        "seed": args.seed,
    }
    if rank == 0:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "architecture_id": ARCHITECTURE_ID_MINI, "distributed_backend": "fsdp_full_shard" if world_size > 1 else "none",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "world_size": world_size, "per_rank_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "steps": args.steps, "seed": args.seed,
            "fold": args.fold, "fold_count": cv_split["fold_count"],
            "metadata": str(Path(args.metadata).resolve()),
            "structure_root": str(Path(args.structure_root).resolve()),
            "split": str(Path(args.split).resolve()),
            "source_split": str(Path(args.source_split).resolve()),
            "cv_split_manifest_sha256": cv_split["manifest_sha256"],
            "source_split_manifest_sha256": source_split["manifest_sha256"],
            "locked_test_record_count": cv_split["locked_test_record_count"],
            "locked_test_loaded_during_training": False,
            "train_requested_record_count": selected_fold["train_record_count"],
            "validation_requested_record_count": selected_fold["val_record_count"],
            "train_qualified_record_count": len(dataset.qualified_record_ids),
            "validation_qualified_record_count": len(validation_dataset.qualified_record_ids),
            "train_missing_record_ids": list(dataset.missing_record_ids),
            "validation_missing_record_ids": list(validation_dataset.missing_record_ids),
            "qualified_example_count": len(dataset), "validation_qualified_example_count": len(validation_dataset),
            "rejected_example_count": len(dataset.rejections),
            "validation_rejected_example_count": len(validation_dataset.rejections),
            "dataset_mapping_sha256": dataset.mapping_sha256,
            "validation_mapping_sha256": validation_dataset.mapping_sha256,
            "validation_every": args.validate_every,
            "validation_max_batches": args.validation_max_batches,
            "validation_dropped_for_equal_ranks": (
                len(validation_dataset) - validation_sampler.total_size if validation_sampler is not None else 0
            ),
            "source_commit": source_commit,
            "preflight_sha256": preflight_sha256,
            "preflight": str(Path(args.preflight).resolve()),
            "command": [sys.executable, *sys.argv],
            "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if device.type == "cuda" else [],
        }
        _write_json(run_dir / "run_manifest.json", manifest)
        (run_dir / "cv_split.json").write_text(Path(args.split).read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "source_split.json").write_text(Path(args.source_split).read_text(encoding="utf-8"), encoding="utf-8")
        (run_dir / "metrics.jsonl").touch()
        _write_json(run_dir / "train_status.json", {"status": "RUNNING", "step": 0, **manifest})
        def mark_incomplete_run():
            if not run_state["completed"]:
                _write_json(run_dir / "train_status.json", {
                    "status": "FAILED", "step": run_state["step"],
                    "world_size": world_size, "reason": "process_exit_without_completion",
                })
        atexit.register(mark_incomplete_run)
        def handle_termination(signum, _frame):
            if not run_state["completed"]:
                _write_json(run_dir / "train_status.json", {
                    "status": "FAILED", "step": run_state["step"],
                    "world_size": world_size, "reason": f"terminated_by_signal_{signum}",
                })
                run_state["completed"] = True
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, handle_termination)
        signal.signal(signal.SIGINT, handle_termination)
    if world_size > 1:
        dist.barrier()
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        run_state["step"] = step
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        output, B = _forward_batch(
            system, batch, device=device, generator=generator, rank=rank,
            step=step, phase="train",
        )
        optimizer.zero_grad(set_to_none=True)
        output.total.backward()
        if isinstance(system, FSDP):
            system.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            reduced, global_samples = _reduce_metrics(
                (output.total, output.core, output.sidechain, output.refine, output.viability), B, device, world_size,
            )
            row = {
                "event": "train", "architecture_id": ARCHITECTURE_ID_MINI,
                "distributed_backend": "fsdp_full_shard" if world_size > 1 else "none",
                "source_commit": source_commit, "preflight_sha256": preflight_sha256,
                "dataset_mapping_sha256": dataset.mapping_sha256, "seed": args.seed,
                "cv_split_manifest_sha256": cv_split["manifest_sha256"],
                "source_split_manifest_sha256": source_split["manifest_sha256"],
                "fold": args.fold,
                "step": step, "world_size": world_size,
                "global_sample_count": global_samples,
                "total": reduced[0], "core": reduced[1], "sidechain": reduced[2],
                "refine": reduced[3], "viability": reduced[4],
            }
            numeric_metric_keys = ("total", "core", "sidechain", "refine", "viability")
            if not all(torch.isfinite(torch.tensor(row[key])) for key in numeric_metric_keys):
                raise RuntimeError("non-finite Mini training metric")
            if rank == 0:
                encoded = json.dumps(row, sort_keys=True, allow_nan=False)
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(encoded + "\n"); handle.flush()
                _write_json(run_dir / "train_status.json", {"status": "RUNNING", **row})
                print(encoded, flush=True)
        if step == 1 or step % args.validate_every == 0:
            validation_values, validation_samples = _validate(
                system, validation_loader, device=device, rank=rank, world_size=world_size,
                seed=args.seed, fold=args.fold, step=step,
                max_batches=args.validation_max_batches,
            )
            validation_row = {
                "event": "validation", "architecture_id": ARCHITECTURE_ID_MINI,
                "distributed_backend": "fsdp_full_shard" if world_size > 1 else "none",
                "source_commit": source_commit, "preflight_sha256": preflight_sha256,
                "dataset_mapping_sha256": validation_dataset.mapping_sha256,
                "cv_split_manifest_sha256": cv_split["manifest_sha256"],
                "source_split_manifest_sha256": source_split["manifest_sha256"],
                "seed": args.seed, "fold": args.fold, "step": step,
                "world_size": world_size, "global_sample_count": validation_samples,
                "total": validation_values[0], "core": validation_values[1],
                "sidechain": validation_values[2], "refine": validation_values[3],
                "viability": validation_values[4],
                "locked_test_loaded": False,
            }
            if rank == 0:
                encoded = json.dumps(validation_row, sort_keys=True, allow_nan=False)
                with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(encoded + "\n"); handle.flush()
                print(encoded, flush=True)
        if step % args.save_every == 0:
            _save_checkpoint(
                run_dir / f"checkpoint-{step:08d}.pt", system=system, optimizer=optimizer,
                step=step, report=report, world_size=world_size, rank=rank,
                provenance=provenance,
            )
    _save_checkpoint(
        run_dir / "checkpoint-final.pt", system=system, optimizer=optimizer,
        step=args.steps, report=report, world_size=world_size, rank=rank,
        provenance=provenance,
    )
    if rank == 0:
        _write_json(run_dir / "train_status.json", {"status": "COMPLETE", "step": args.steps, "world_size": world_size})
        run_state["completed"] = True
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
