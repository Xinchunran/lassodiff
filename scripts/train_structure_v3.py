#!/usr/bin/env python3
"""Four-GPU FSDP structure-phase trainer for cached OpenDDE LassoDiff V3."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardedStateDictConfig, StateDictType
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lassodiff.architecture_contract_v3 import (
    checkpoint_manifest_v3, load_architecture_config_v3, sha256_json_file,
)
from lassodiff.candidate_objective import candidate_marginal_loss
from lassodiff.data.lassopred_lmdb import LassoPredLMDBDataset
from lassodiff.data.family_split import validate_family_split_manifest
from lassodiff.data.opendde_cached import OpenDDECachedDataset, collate_lassodiff_v3
from lassodiff.evaluation_v2 import evaluate_generated_structures
from lassodiff.model_v3 import LassoDiffOpenDDEV3
from lassodiff.opendde_bridge.cache import OpenDDEReasoningCache, ReasoningCacheIdentity
from lassodiff.opendde_bridge.reasoner import CacheOnlyOpenDDEReasoner
from lassodiff.preflight_v3 import build_v3_optimizer
from lassodiff.sampler_v3 import sample_rectified_flow_v3
from lassodiff.topology_adapter import CandidateBatch
from lassodiff.topology_checker import strict_topology_check
from lassodiff.topology_scoring import score_candidates
from lassodiff.threading_truth import validate_threading_alignment_report
from lassodiff.topology_objective import TopologyLossWeights, topology_supervised_candidate_loss


def _rank_info():
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])


def _append(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _status(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _apply_diffusion_checkpointing(model):
    block_ids = {id(block) for block in model.diffusion.blocks}
    wrapper = lambda module: checkpoint_wrapper(
        module, checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=False,
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=wrapper, check_fn=lambda module: id(module) in block_ids,
    )


def _move_state(state, device):
    return state.to(device, dtype=torch.float32)


def _move(batch, device):
    output = {}
    for key, value in batch.items():
        output[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    output["reasoning_state"] = _move_state(batch["reasoning_state"], device)
    return output


def _candidates(batch):
    valid = batch["candidate_mask"] & batch.get("target_valid", batch["candidate_mask"])
    return CandidateBatch(
        batch["k"], batch["p"], batch["acceptor_index"], batch["candidate_prior"],
        valid, acceptor_type=batch["acceptor_type"],
    )


def _identity_builder(config):
    def build(sequence):
        return ReasoningCacheIdentity(
            sequence, config.reasoning.opendde_commit, config.reasoning.checkpoint_sha256,
            config.reasoning.cache_schema_version, config.reasoning.use_msa,
            config.reasoning.use_template, config.reasoning.dtype, config.reasoning.n_cycle,
        )
    return build


def _candidate_mse(velocity, target_velocity, atom_mask, candidate_mask):
    weight = atom_mask[..., None].to(velocity.dtype)
    squared = (velocity - target_velocity).square() * weight
    denominator = weight.sum(dim=(-3, -2, -1)).clamp(min=1.0) * 3.0
    loss = squared.sum(dim=(-3, -2, -1)) / denominator
    return torch.where(candidate_mask, loss, torch.zeros_like(loss))


def _topology_loss_weights(config):
    structure = config.structure
    return TopologyLossWeights(
        flow=structure.flow_loss_weight,
        bond=structure.bond_loss_weight,
        iso_distance=structure.iso_distance_loss_weight,
        iso_angle=structure.iso_angle_loss_weight,
        iso_plane=structure.iso_plane_loss_weight,
        threading=structure.threading_loss_weight,
    )


_LOSS_COMPONENTS = ("flow", "bond", "iso_distance", "iso_angle", "iso_plane", "threading")


def _center_targets(coords, atom_mask):
    weight = atom_mask[..., None].to(coords.dtype)
    centroid = (coords * weight).sum(dim=(-3, -2), keepdim=True) / weight.sum(
        dim=(-3, -2), keepdim=True
    ).clamp(min=1.0)
    return (coords - centroid) * weight


def _candidate_diversity(coords, candidate_mask, token_mask):
    ca = coords[..., 1, :].float()
    weight=token_mask[:,None,:,None].to(ca.dtype)
    ca = ca - (ca*weight).sum(dim=-2,keepdim=True)/weight.sum(dim=-2,keepdim=True).clamp_min(1)
    values = []
    for left in range(ca.shape[1]):
        for right in range(left + 1, ca.shape[1]):
            valid = candidate_mask[:, left] & candidate_mask[:, right]
            if valid.any():
                squared=(ca[valid,left]-ca[valid,right]).square().sum(-1)
                values.append(torch.sqrt((squared*token_mask[valid]).sum(-1)/token_mask[valid].sum(-1).clamp_min(1)))
    return torch.cat(values).mean() if values else coords.new_zeros(())


def _plug_tail_ring_geometry(coords, candidates, token_mask):
    ca = coords[..., 1, :].float(); plug_values=[]; tail_values=[]
    for b in range(ca.shape[0]):
        for m in range(ca.shape[1]):
            if not bool(candidates.candidate_mask[b,m]):continue
            k,p=int(candidates.k[b,m]),int(candidates.p[b,m]);length=int(token_mask[b].sum());ring=ca[b,m,:k+1]
            plug_values.append(torch.cdist(ca[b,m,p:p+1],ring).min())
            if p+1<length:tail_values.append(torch.cdist(ca[b,m,p+1:length],ring).min())
    zero=coords.new_zeros(())
    return (torch.stack(plug_values).mean() if plug_values else zero,
            torch.stack(tail_values).mean() if tail_values else zero)


@torch.no_grad()
def _teacher_validation(model, loader, device, loss_weights, max_batches=16):
    model.eval()
    totals = torch.zeros(4 + len(_LOSS_COMPONENTS), dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(2903 + dist.get_rank())
    for index, raw in enumerate(loader):
        if index >= max_batches:
            break
        batch = _move(raw, device)
        state, target = batch["reasoning_state"], _center_targets(batch["coords"], batch["atom_mask"])
        candidates = _candidates(batch)
        noise = torch.randn(target.shape, generator=generator, device=device)
        t = torch.rand((target.shape[0],), generator=generator, device=device)
        x_t = (1 - t[:, None, None, None, None]) * noise + t[:, None, None, None, None] * target
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(state, candidates, x_t, t, batch["atom_mask"])
            breakdown = topology_supervised_candidate_loss(
                output.velocity, target - noise, x_t, t, target, state.token_mask,
                batch["atom_mask"], candidates, loss_weights,
            )
            loss, posterior, _ = candidate_marginal_loss(
                breakdown.total, batch["candidate_prior"], candidates.candidate_mask,
            )
        valid = candidates.candidate_mask
        entropy = -(posterior * posterior.clamp_min(1e-12).log()).sum(-1)
        totals += torch.tensor([
            float(loss) * target.shape[0], float(entropy.sum()), target.shape[0], int(valid.sum())
        ] + [float(getattr(breakdown, name)[valid].sum()) for name in _LOSS_COMPONENTS],
            dtype=torch.float64, device=device)
    dist.all_reduce(totals)
    model.train()
    result = {
        "loss": float(totals[0] / totals[2].clamp_min(1)),
        "posterior_entropy": float(totals[1] / totals[2].clamp_min(1)),
        "sample_count": int(totals[2]), "candidate_count": int(totals[3]),
    }
    for index, name in enumerate(_LOSS_COMPONENTS):
        result[f"{name}_loss"] = float(totals[4 + index] / totals[3].clamp_min(1))
    return result


@torch.no_grad()
def _rollout_validation(model, loader, device, sampling_steps, max_batches=2):
    model.eval()
    metric_keys = ("ca_rmsd", "backbone_rmsd", "ca_lddt", "iso_distance_mae", "link_class_accuracy", "topology_match_rate", "bond_length_mae")
    topology_keys = (
        "iso_distance", "iso_angle_degrees", "iso_plane_distance",
        "iso_distance_valid", "iso_angle_valid", "iso_plane_valid",
        "checker_valid", "crossing_count", "signed_crossing", "threading_class", "threading_confidence",
        "ring_closed", "plug_retained", "tail_clearance", "small_perturbation_stable",
        "ring_disk_crossing", "threading_success", "clash_rate", "topology_pass",
    )
    auxiliary_keys = ("candidate_diversity", "plug_ring_min_distance", "tail_ring_min_distance", "gauss_link_abs", "sampler_stability", "score_hard_valid", "candidate_score")
    totals = torch.zeros(len(metric_keys) + len(topology_keys) + len(auxiliary_keys) + 2, dtype=torch.float64, device=device)
    generator = torch.Generator(device=device).manual_seed(3907 + dist.get_rank())
    for index, raw in enumerate(loader):
        if index >= max_batches:
            break
        batch = _move(raw, device)
        state, candidates = batch["reasoning_state"], _candidates(batch)
        initial = torch.randn(batch["coords"].shape, generator=generator, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sampled = sample_rectified_flow_v3(
                model, state, candidates, batch["atom_mask"], steps=sampling_steps,
                initial_coordinates=initial,
            ).coordinates
        metrics = evaluate_generated_structures(
            sampled.float(), batch["coords"], batch["token_mask"], batch["atom_mask"], candidates,
        )
        topology = strict_topology_check(
            sampled.float(), batch["atom_mask"], candidates, token_mask=batch["token_mask"],
        )
        candidate_score = score_candidates(topology)
        count = int(candidates.candidate_mask.sum())
        for position, key in enumerate(metric_keys):
            totals[position] += float(metrics[key]) * count
        valid = candidates.candidate_mask.float()
        offset = len(metric_keys)
        for position, key in enumerate(topology_keys):
            value = getattr(topology, key).float()
            totals[offset + position] += float((value * valid).sum())
        plug_ring,tail_ring=_plug_tail_ring_geometry(sampled,candidates,batch["token_mask"])
        finite_scores = torch.where(candidate_score.hard_valid, candidate_score.score, torch.zeros_like(candidate_score.score))
        auxiliary=(
            _candidate_diversity(sampled,candidates.candidate_mask,batch["token_mask"]),plug_ring,tail_ring,
            (topology.gauss_link_value.abs()*valid).sum()/valid.sum().clamp_min(1),sampled.new_tensor(1.0),
            candidate_score.hard_valid.float().mul(valid).sum()/valid.sum().clamp_min(1),
            (finite_scores*valid).sum()/valid.sum().clamp_min(1),
        )
        offset += len(topology_keys)
        for position,value in enumerate(auxiliary):totals[offset+position] += float(value)*count
        totals[-2] += count
        totals[-1] += batch["coords"].shape[0]
    dist.all_reduce(totals)
    model.train()
    count = totals[-2].clamp_min(1)
    result = {key: float(totals[index] / count) for index, key in enumerate(metric_keys)}
    result.update({key: float(totals[len(metric_keys) + index] / count) for index, key in enumerate(topology_keys)})
    auxiliary_offset=len(metric_keys)+len(topology_keys)
    result.update({key:float(totals[auxiliary_offset+index]/count) for index,key in enumerate(auxiliary_keys)})
    result.update({"candidate_count": int(totals[-2]), "sample_count": int(totals[-1]), "sampling_steps": sampling_steps})
    return result


def _save(model, optimizer, directory, step, manifest, rank):
    directory.mkdir(parents=True, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                              ShardedStateDictConfig(offload_to_cpu=True)):
        state = {"model": model.state_dict(), "optimizer": FSDP.optim_state_dict(model, optimizer),
                 "step": step, "architecture_manifest": manifest}
    torch.save(state, directory / f"checkpoint.rank{rank:04d}.pt")
    if rank == 0:
        (directory / "checkpoint.index.json").write_text(json.dumps({
            "step": step, "world_size": dist.get_world_size(), "format": "fsdp_sharded_v1",
            "architecture_id": "lassodiff_opendde_v3",
        }, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--threading-report", required=True)
    parser.add_argument("--startup-gate", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--rollout-every", type=int, default=500)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--topology-gate-step", type=int, default=2000)
    parser.add_argument("--min-iso-distance-valid", type=float, default=0.01)
    parser.add_argument("--min-topology-pass", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    rank, world_size, local_rank = _rank_info()
    if world_size != 4 or not torch.cuda.is_available():
        raise RuntimeError("formal V3 structure training requires exactly four CUDA ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    try:
        config = load_architecture_config_v3(args.config)
        preflight = json.loads(Path(args.preflight).read_text(encoding="utf-8"))
        if preflight.get("status") != "PASS" or preflight.get("trace", {}).get("reasoner_called") != 1:
            raise RuntimeError("formal V3 training requires a PASS preflight with real reasoner route")
        if preflight.get("reasoning_state", {}).get("checkpoint_sha256") != config.reasoning.checkpoint_sha256:
            raise RuntimeError("preflight checkpoint does not match training config")
        split = json.loads(Path(args.split).read_text(encoding="utf-8"))
        cache_manifest = json.loads(Path(args.cache_manifest).read_text(encoding="utf-8"))
        threading_report = json.loads(Path(args.threading_report).read_text(encoding="utf-8"))
        validate_threading_alignment_report(threading_report)
        startup_gate = json.loads(Path(args.startup_gate).read_text(encoding="utf-8"))
        if startup_gate.get("status") != "PASS":
            raise RuntimeError("formal V3 training requires a PASS startup gate")
        if startup_gate.get("split_manifest_sha256") != split.get("manifest_sha256"):
            raise RuntimeError("startup gate split hash does not match training split")
        if startup_gate.get("threading_alignment_report_sha256") != threading_report.get("report_sha256"):
            raise RuntimeError("startup gate threading report hash does not match report")
        expected_cache = ("PASS", split.get("unique_sequence_count"), config.reasoning.checkpoint_sha256, config.reasoning.opendde_commit,
                          config.reasoning.n_cycle, config.reasoning.cache_schema_version)
        got_cache = (cache_manifest.get("status"), cache_manifest.get("sequence_count"),
                     cache_manifest.get("checkpoint_sha256"), cache_manifest.get("opendde_commit"),
                     cache_manifest.get("n_cycle"), cache_manifest.get("feature_schema_version"))
        if got_cache != expected_cache:
            raise RuntimeError(f"full OpenDDE cache manifest mismatch: expected={expected_cache}, got={got_cache}")
        validate_family_split_manifest(split)
        split_sha = sha256_json_file(args.split)
        manifest = checkpoint_manifest_v3(config, world_size, split_sha)
        loss_weights = _topology_loss_weights(config)
        run_dir = Path(args.run_dir)
        if rank == 0:
            if run_dir.exists():
                unexpected = [path.name for path in run_dir.iterdir() if path.name != "launcher.log"]
                if unexpected:
                    raise FileExistsError(f"V3 run directory is not empty: {unexpected[:5]}")
            else:
                run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "architecture_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            (run_dir / "config.yaml").write_text(Path(args.config).read_text())
            (run_dir / "split.json").write_text(Path(args.split).read_text())
            (run_dir / "preflight.json").write_text(Path(args.preflight).read_text())
            (run_dir / "cache_manifest.json").write_text(Path(args.cache_manifest).read_text())
            (run_dir / "threading_alignment.json").write_text(Path(args.threading_report).read_text())
            (run_dir / "startup_gate.json").write_text(Path(args.startup_gate).read_text())
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
            dirty = subprocess.check_output(["git", "status", "--short"], cwd=_ROOT, text=True).splitlines()
            (run_dir / "source_manifest.json").write_text(json.dumps({
                "git_revision": revision, "dirty": bool(dirty), "dirty_paths": dirty,
                "torch_version": torch.__version__, "cuda": torch.version.cuda,
                "prior_source": "teacher_prior_structure_phase_only",
            }, indent=2) + "\n")
        dist.barrier()
        identity = _identity_builder(config)
        base = LassoPredLMDBDataset(
            args.dataset, split["train"], seed=args.seed,
            target_policy=config.structure.target_policy,
        )
        validation_base = LassoPredLMDBDataset(
            args.dataset, split["val"], seed=args.seed + 1,
            target_policy=config.structure.target_policy,
        )
        cache = OpenDDEReasoningCache(config.reasoning.cache_dir)
        dataset = OpenDDECachedDataset(base, cache, identity)
        validation = OpenDDECachedDataset(validation_base, cache, identity)
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=True, drop_last=True, seed=args.seed)
        validation_sampler = DistributedSampler(validation, world_size, rank, shuffle=False, drop_last=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True, collate_fn=collate_lassodiff_v3)
        validation_loader = DataLoader(validation, batch_size=max(1, args.batch_size // 2), sampler=validation_sampler,
                                       num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                       collate_fn=collate_lassodiff_v3)
        reasoner = CacheOnlyOpenDDEReasoner(config.reasoning.checkpoint_sha256, config.reasoning.opendde_commit)
        model = LassoDiffOpenDDEV3(
            reasoner, 384, 384, c_s=config.model.c_s, c_z=config.model.c_z, c_a=config.model.c_a,
            n_heads=config.model.n_heads, diffusion_blocks=config.model.diffusion_blocks,
            max_candidates=config.topology.max_candidates,
            structure_gradient_scale=config.structure.reasoning_gradient_scale,
        ).to(device)
        trainable_full = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if trainable_full != config.model.expected_trainable_numel:
            raise RuntimeError(
                f"V3 trainable parameter manifest mismatch: expected={config.model.expected_trainable_numel}, got={trainable_full}"
            )
        _apply_diffusion_checkpointing(model)
        mixed = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16)
        model = FSDP(model, device_id=device, mixed_precision=mixed, use_orig_params=True)
        optimizer = build_v3_optimizer(model.module, config)
        metric_identity = {
            "architecture_id": config.architecture_id,
            "metrics_schema_version": 3,
            "checkpoint_sha256": config.reasoning.checkpoint_sha256,
            "opendde_checkpoint_sha256": config.reasoning.checkpoint_sha256,
            "split_manifest_sha256": split_sha,
            "seed": args.seed,
            "target_policy": config.structure.target_policy,
            "threading_checker_version": threading_report["checker_version"],
            "threading_alignment_report_sha256": threading_report["report_sha256"],
        }
        if rank == 0:
            start_payload = {
                **metric_identity, "event": "start", "world_size": world_size,
                "batch_size_per_gpu": args.batch_size, "global_batch_size": args.batch_size * world_size,
                "max_steps": args.max_steps, "trainable_parameters": trainable_full,
                "projection_used": False, "topology_guidance_used": False,
                "activation_checkpointing": "non_reentrant_per_diffusion_block",
                "loss_weights": vars(loss_weights),
                "source_train_records": base.source_record_count,
                "qualified_train_records": base.qualified_record_count,
                "source_validation_records": validation_base.source_record_count,
                "qualified_validation_records": validation_base.qualified_record_count,
                "topology_gate_step": args.topology_gate_step,
                "min_iso_distance_valid": args.min_iso_distance_valid,
                "min_topology_pass": args.min_topology_pass,
            }
            _append(run_dir / "metrics.jsonl", start_payload)
            _status(run_dir / "train_status.json", {"status": "RUNNING", "step": 0, **start_payload})
        generator = torch.Generator(device=device).manual_seed(args.seed + rank * 100003)
        step, epoch = 0, 0
        while step < args.max_steps:
            sampler.set_epoch(epoch)
            dataset.set_epoch(epoch)
            for raw in loader:
                batch = _move(raw, device)
                state, target, candidates = (
                    batch["reasoning_state"], _center_targets(batch["coords"], batch["atom_mask"]), _candidates(batch)
                )
                noise = torch.randn(target.shape, generator=generator, device=device)
                t = torch.rand((target.shape[0],), generator=generator, device=device)
                x_t = (1 - t[:, None, None, None, None]) * noise + t[:, None, None, None, None] * target
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(state, candidates, x_t, t, batch["atom_mask"])
                    breakdown = topology_supervised_candidate_loss(
                        output.velocity, target - noise, x_t, t, target, state.token_mask,
                        batch["atom_mask"], candidates, loss_weights,
                    )
                    loss, posterior, normalized_prior = candidate_marginal_loss(
                        breakdown.total, batch["candidate_prior"].detach(), candidates.candidate_mask,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                if step % 10 == 0 or step == 1:
                    valid = candidates.candidate_mask
                    values = torch.tensor([float(loss), float(breakdown.total[valid].mean()),
                                           float((-(posterior * posterior.clamp_min(1e-12).log()).sum(-1)).mean())]
                                          + [float(getattr(breakdown, name)[valid].mean()) for name in _LOSS_COMPONENTS],
                                          device=device, dtype=torch.float64)
                    dist.all_reduce(values)
                    if rank == 0:
                        train_payload = {
                            **metric_identity, "event": "metrics", "split": "train", "step": step,
                            "loss": float(values[0] / world_size), "candidate_loss": float(values[1] / world_size),
                            "posterior_entropy": float(values[2] / world_size),
                            "max_mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
                            "prior_source": "teacher_prior_structure_phase_only", "time": time.time(),
                            "sample_count": args.batch_size * world_size,
                        }
                        for index, name in enumerate(_LOSS_COMPONENTS):
                            train_payload[f"{name}_loss"] = float(values[3 + index] / world_size)
                        _append(run_dir / "metrics.jsonl", train_payload)
                        _status(run_dir / "train_status.json", {"status": "RUNNING", **train_payload})
                if step % args.validation_every == 0:
                    validation_metrics = _teacher_validation(model, validation_loader, device, loss_weights)
                    if rank == 0:
                        _append(run_dir / "metrics.jsonl", {**metric_identity, "event": "metrics", "split": "validation_teacher",
                                                            "step": step, **validation_metrics})
                gate_failure = None
                if step % args.rollout_every == 0:
                    rollout = _rollout_validation(model, validation_loader, device, args.sampling_steps)
                    if rank == 0:
                        _append(run_dir / "metrics.jsonl", {**metric_identity, "event": "metrics", "split": "validation_rollout",
                                                            "step": step, "projection_used": False,
                                                            "topology_guidance_used": False, **rollout})
                    if step == args.topology_gate_step and (
                        rollout["iso_distance_valid"] < args.min_iso_distance_valid
                        or rollout["topology_pass"] < args.min_topology_pass
                    ):
                        gate_failure = (
                            f"topology gate failed at step {step}: iso_distance_valid="
                            f"{rollout['iso_distance_valid']:.6f}, topology_pass={rollout['topology_pass']:.6f}"
                        )
                if step % args.save_every == 0:
                    _save(model, optimizer, run_dir / f"checkpoint-{step:08d}", step, manifest, rank)
                if gate_failure is not None:
                    if rank == 0:
                        _append(run_dir / "metrics.jsonl", {**metric_identity, "event": "failed", "step": step,
                                                            "reason": gate_failure, "time": time.time()})
                        _status(run_dir / "train_status.json", {"status": "FAILED", "step": step,
                                                                 "reason": gate_failure, "time": time.time()})
                    raise RuntimeError(gate_failure)
                if step >= args.max_steps:
                    break
            epoch += 1
        _save(model, optimizer, run_dir / "checkpoint-final", step, manifest, rank)
        if rank == 0:
            _append(run_dir / "metrics.jsonl", {**metric_identity, "event": "complete", "step": step, "time": time.time()})
            _status(run_dir / "train_status.json", {"status": "COMPLETE", "step": step, "time": time.time()})
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
