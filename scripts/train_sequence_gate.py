#!/usr/bin/env python3
"""Four-rank FSDP trainer for verified positive/background sequence gating."""
from __future__ import annotations

import argparse, json, os, random, sys, time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardedStateDictConfig, FullStateDictConfig, StateDictType
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path: sys.path.insert(0, str(_ROOT))

from lassodiff.architecture_contract_v3 import load_architecture_config_v3
from lassodiff.data.sequence_cached import CachedSequenceDataset, collate_sequence_gate
from lassodiff.data.sequence_v3 import load_sequence_manifest
from lassodiff.opendde_bridge.cache import OpenDDEReasoningCache, ReasoningCacheIdentity
from lassodiff.sequence_metrics import calibrate_gate_thresholds, sequence_classification_metrics
from lassodiff.sequence_model_v3 import CachedOpenDDESequenceModel
from lassodiff.sequence_objective import sequence_gate_objective


def _append(path, row):
    with path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(row, sort_keys=True) + "\n")


def _identity(config, sequence):
    return ReasoningCacheIdentity(sequence, config.reasoning.opendde_commit, config.reasoning.checkpoint_sha256,
        config.reasoning.cache_schema_version, False, False, config.reasoning.dtype, config.reasoning.n_cycle)


def _move(batch, device):
    output = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    output["reasoning_state"] = batch["reasoning_state"].to(device, dtype=torch.float32)
    return output


def _save(model, optimizer, directory, step, rank, config):
    directory.mkdir(parents=True, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                              ShardedStateDictConfig(offload_to_cpu=True)):
        state={"model":model.state_dict(),"optimizer":FSDP.optim_state_dict(model,optimizer),"step":step,
               "stage_manifest":{"architecture_id":config.architecture_id,"schema_version":3,
                   "stage":"sequence_gate","required_modules":["reasoning_adapter","sequence_gate"],
                   "opendde_checkpoint_sha256":config.reasoning.checkpoint_sha256,
                   "opendde_commit":config.reasoning.opendde_commit,"allow_fallback":False}}
    torch.save(state,directory/f"checkpoint.rank{rank:04d}.pt")
    if rank==0:(directory/"checkpoint.index.json").write_text(json.dumps({"step":step,"world_size":dist.get_world_size(),
        "format":"fsdp_sharded_v1","stage":"sequence_gate"},indent=2)+"\n")


def _save_full_gate(model, path, step, rank, config):
    with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,
                              FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        state=model.state_dict()
    if rank==0:
        torch.save({"model":state,"step":step,"stage_manifest":{"architecture_id":config.architecture_id,
            "schema_version":3,"stage":"sequence_gate","opendde_checkpoint_sha256":config.reasoning.checkpoint_sha256,
            "opendde_commit":config.reasoning.opendde_commit,"allow_fallback":False}},path)


@torch.no_grad()
def _validate(model, loader, device, config):
    model.eval(); probability, label, ood = [], [], []
    for raw in loader:
        batch = _move(raw, device)
        with torch.autocast("cuda", dtype=torch.bfloat16): out = model(batch["reasoning_state"])
        probability.append((1 - torch.sigmoid(out.no_lasso_logit)).float())
        label.append(batch["label"]); ood.append(batch["ood_target"].bool())
    local = torch.cat(probability), torch.cat(label), torch.cat(ood)
    gathered = [[None] * dist.get_world_size() for _ in range(3)]
    for index, tensor in enumerate(local): dist.all_gather_object(gathered[index], tensor.cpu())
    probability = torch.cat(gathered[0]); label = torch.cat(gathered[1]); ood = torch.cat(gathered[2])
    thresholds = calibrate_gate_thresholds(probability, label, target_fpr=config.sequence_gate.target_validation_fpr,
                                           target_recall=.8)
    decisions = ["LASSO_PLAUSIBLE" if value >= thresholds["accept_threshold"] else
                 "NON_LASSO" if value <= thresholds["reject_threshold"] else "ABSTAIN"
                 for value in probability.tolist()]
    metrics = sequence_classification_metrics(probability, label, decisions, ood_mask=ood)
    model.train(); return {**thresholds, **metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--preflight", required=True); parser.add_argument("--run-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=2); parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 4 or not torch.cuda.is_available(): raise RuntimeError("formal sequence gate training requires four CUDA ranks")
    torch.cuda.set_device(local_rank); device=torch.device("cuda", local_rank); dist.init_process_group("nccl",device_id=device)
    try:
        config=load_architecture_config_v3(args.config)
        preflight=json.loads(Path(args.preflight).read_text())
        if preflight.get("status") != "PASS" or preflight.get("trace",{}).get("reasoner_called") != 1:
            raise RuntimeError("sequence training requires real OpenDDE PASS preflight")
        payload, examples=load_sequence_manifest(args.manifest)
        kinds={row.kind for row in examples}
        required={"positive","background","composition_shuffle","hard_mutant","wrong_topology","random"}
        if not required <= kinds: raise RuntimeError(f"sequence release dataset lacks negative strata: {sorted(required-kinds)}")
        by_id={row.example_id:row for row in examples}; split=payload["split"]
        cache=OpenDDEReasoningCache(config.reasoning.cache_dir)
        make=lambda ids: CachedSequenceDataset([by_id[item] for item in ids], cache, lambda seq:_identity(config,seq))
        train, validation=make(split["train"]), make(split["val"])
        sampler=DistributedSampler(train,world,rank,shuffle=True,drop_last=True,seed=args.seed)
        val_sampler=DistributedSampler(validation,world,rank,shuffle=False,drop_last=False)
        loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=args.num_workers,
                          collate_fn=collate_sequence_gate,drop_last=True,pin_memory=True)
        val_loader=DataLoader(validation,batch_size=args.batch_size,sampler=val_sampler,num_workers=args.num_workers,
                              collate_fn=collate_sequence_gate,drop_last=False,pin_memory=True)
        module=CachedOpenDDESequenceModel(c_s=config.model.c_s,c_z=config.model.c_z,
                                          max_candidates=config.topology.max_candidates).to(device)
        model=FSDP(module,device_id=device,use_orig_params=True,mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16))
        optimizer=torch.optim.AdamW([
            {"params":model.module.reasoning_adapter.parameters(),"lr":config.training.adapter_lr},
            {"params":model.module.sequence_gate.parameters(),"lr":config.training.gate_lr}],weight_decay=.01)
        run=Path(args.run_dir)
        if rank==0:
            run.mkdir(parents=True,exist_ok=False); (run/"manifest.json").write_text(Path(args.manifest).read_text())
            (run/"preflight.json").write_text(Path(args.preflight).read_text())
            _append(run/"metrics.jsonl",{"event":"start","architecture_id":config.architecture_id,
                "global_batch_size":args.batch_size*world,"max_steps":args.max_steps,"metrics_schema_version":1})
        dist.barrier(); step=epoch=0
        while step<args.max_steps:
            sampler.set_epoch(epoch)
            for raw in loader:
                batch=_move(raw,device); optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    assessment=model(batch["reasoning_state"])
                    losses=sequence_gate_objective(assessment,label=batch["label"],token_mask=batch["reasoning_state"].token_mask,
                        acceptor_index=batch["acceptor_index"],plug_index=batch["plug_index"],teacher_prior=batch["teacher_prior"],
                        candidate_mask=batch["candidate_mask"],ood_target=batch["ood_target"])
                losses["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); step+=1
                if step%10==0:
                    values=torch.stack([losses[key].detach().float() for key in ("total","gate","hypothesis","position","ood")])
                    dist.all_reduce(values)
                    if rank==0:_append(run/"metrics.jsonl",{"event":"metrics","split":"train","step":step,
                        **{key:float(values[i]/world) for i,key in enumerate(("loss","gate_loss","hypothesis_loss","position_loss","ood_loss"))},"time":time.time()})
                if step%250==0:
                    metrics=_validate(model,val_loader,device,config)
                    if rank==0:_append(run/"metrics.jsonl",{"event":"metrics","split":"validation","step":step,**metrics})
                if step%500==0:_save(model,optimizer,run/f"checkpoint-{step:08d}",step,rank,config)
                if step>=args.max_steps:break
            epoch+=1
        _save(model,optimizer,run/"checkpoint-final",step,rank,config)
        _save_full_gate(model,run/"sequence_gate.full.pt",step,rank,config)
        if rank==0:_append(run/"metrics.jsonl",{"event":"complete","step":step,"time":time.time()})
    finally: dist.destroy_process_group()


if __name__ == "__main__": main()
