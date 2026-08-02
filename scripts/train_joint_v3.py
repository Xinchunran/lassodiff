#!/usr/bin/env python3
"""Controlled P6 joint trainer; requires a verified gate checkpoint and both datasets."""
from __future__ import annotations

import argparse,json,os,sys,time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,MixedPrecision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_ROOT=Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:sys.path.insert(0,str(_ROOT))

from lassodiff.architecture_contract_v3 import checkpoint_manifest_v3,load_architecture_config_v3,sha256_json_file
from lassodiff.candidate_objective import candidate_marginal_loss
from lassodiff.data.lassopred_lmdb import LassoPredLMDBDataset
from lassodiff.data.family_split import validate_family_split_manifest
from lassodiff.data.opendde_cached import OpenDDECachedDataset,collate_lassodiff_v3
from lassodiff.data.sequence_cached import CachedSequenceDataset,collate_sequence_gate
from lassodiff.data.sequence_v3 import load_sequence_manifest
from lassodiff.model_v3 import LassoDiffOpenDDEV3
from lassodiff.opendde_bridge.cache import OpenDDEReasoningCache
from lassodiff.opendde_bridge.reasoner import CacheOnlyOpenDDEReasoner
from lassodiff.preflight_v3 import build_v3_optimizer
from lassodiff.sequence_objective import sequence_gate_objective
from scripts.train_structure_v3 import (_candidates,_identity_builder,_move,_save,_teacher_validation,
                                        _center_targets,_apply_diffusion_checkpointing,_topology_loss_weights)
from lassodiff.topology_objective import topology_supervised_candidate_loss
from scripts.train_sequence_gate import _identity as _sequence_identity,_move as _move_sequence,_validate as _validate_sequence


def _load_gate_strict(model,path,config):
    payload=torch.load(path,map_location="cpu",weights_only=False);manifest=payload.get("stage_manifest",{})
    expected=(config.architecture_id,3,"sequence_gate",config.reasoning.checkpoint_sha256,config.reasoning.opendde_commit,False)
    got=(manifest.get("architecture_id"),manifest.get("schema_version"),manifest.get("stage"),
         manifest.get("opendde_checkpoint_sha256"),manifest.get("opendde_commit"),manifest.get("allow_fallback"))
    if got!=expected:raise RuntimeError(f"gate checkpoint manifest mismatch: {got}")
    state=payload["model"]
    for prefix,module in (("reasoning_adapter.",model.reasoning_adapter),("sequence_gate.",model.sequence_gate)):
        subset={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
        if not subset:raise RuntimeError(f"gate checkpoint has no {prefix} parameters")
        result=module.load_state_dict(subset,strict=True)
        if result.missing_keys or result.unexpected_keys:raise RuntimeError("strict gate subset load failed")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--dataset",required=True)
    parser.add_argument("--split",required=True);parser.add_argument("--sequence-manifest",required=True)
    parser.add_argument("--gate-checkpoint",required=True);parser.add_argument("--preflight",required=True)
    parser.add_argument("--run-dir",required=True);parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--max-steps",type=int,default=10000);parser.add_argument("--num-workers",type=int,default=2)
    parser.add_argument("--seed",type=int,default=1701);args=parser.parse_args()
    rank,world,local=int(os.environ["RANK"]),int(os.environ["WORLD_SIZE"]),int(os.environ["LOCAL_RANK"])
    if world!=4 or not torch.cuda.is_available():raise RuntimeError("joint V3 requires exactly four CUDA ranks")
    torch.cuda.set_device(local);device=torch.device("cuda",local);dist.init_process_group("nccl",device_id=device)
    try:
        config=load_architecture_config_v3(args.config);preflight=json.loads(Path(args.preflight).read_text())
        if preflight.get("status")!="PASS" or preflight.get("trace",{}).get("reasoner_called")!=1:
            raise RuntimeError("joint training requires real PASS preflight")
        split=json.loads(Path(args.split).read_text());split_sha=sha256_json_file(args.split)
        validate_family_split_manifest(split)
        sequence_payload,examples=load_sequence_manifest(args.sequence_manifest);by_id={row.example_id:row for row in examples}
        cache=OpenDDEReasoningCache(config.reasoning.cache_dir);identity=_identity_builder(config)
        structure=OpenDDECachedDataset(LassoPredLMDBDataset(
            args.dataset,split["train"],seed=args.seed,target_policy=config.structure.target_policy),cache,identity)
        sequence=CachedSequenceDataset([by_id[item] for item in sequence_payload["split"]["train"]],cache,
                                       lambda seq:_sequence_identity(config,seq))
        struct_sampler=DistributedSampler(structure,world,rank,shuffle=True,drop_last=True,seed=args.seed)
        seq_sampler=DistributedSampler(sequence,world,rank,shuffle=True,drop_last=True,seed=args.seed+1)
        struct_loader=DataLoader(structure,batch_size=args.batch_size,sampler=struct_sampler,num_workers=args.num_workers,
                                 collate_fn=collate_lassodiff_v3,drop_last=True,pin_memory=True)
        seq_loader=DataLoader(sequence,batch_size=args.batch_size,sampler=seq_sampler,num_workers=args.num_workers,
                              collate_fn=collate_sequence_gate,drop_last=True,pin_memory=True)
        module=LassoDiffOpenDDEV3(CacheOnlyOpenDDEReasoner(config.reasoning.checkpoint_sha256,config.reasoning.opendde_commit),
            384,384,c_s=config.model.c_s,c_z=config.model.c_z,c_a=config.model.c_a,n_heads=config.model.n_heads,
            diffusion_blocks=config.model.diffusion_blocks,max_candidates=config.topology.max_candidates,
            structure_gradient_scale=config.structure.reasoning_gradient_scale).to(device)
        _load_gate_strict(module,args.gate_checkpoint,config)
        _apply_diffusion_checkpointing(module)
        model=FSDP(module,device_id=device,use_orig_params=True,mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16))
        optimizer=build_v3_optimizer(model.module,config);manifest=checkpoint_manifest_v3(config,world,split_sha)
        topology_weights=_topology_loss_weights(config)
        run=Path(args.run_dir)
        if rank==0:
            run.mkdir(parents=True,exist_ok=False);(run/"architecture_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
            (run/"sequence_manifest.json").write_text(Path(args.sequence_manifest).read_text())
            (run/"metrics.jsonl").write_text(json.dumps({"event":"start","stage":"joint","global_batch_size":args.batch_size*world,
                "candidate_prior_detach":True,"reasoning_gradient_scale":config.structure.reasoning_gradient_scale})+"\n")
        dist.barrier();struct_iter=iter(struct_loader);seq_iter=iter(seq_loader);step=0
        generator=torch.Generator(device=device).manual_seed(args.seed+rank*100003)
        while step<args.max_steps:
            try:sraw=next(struct_iter)
            except StopIteration:struct_iter=iter(struct_loader);sraw=next(struct_iter)
            try:qraw=next(seq_iter)
            except StopIteration:seq_iter=iter(seq_loader);qraw=next(seq_iter)
            sbatch=_move(sraw,device);qbatch=_move_sequence(qraw,device);candidates=_candidates(sbatch)
            target=_center_targets(sbatch["coords"],sbatch["atom_mask"]);noise=torch.randn(target.shape,generator=generator,device=device)
            t=torch.rand((target.shape[0],),generator=generator,device=device)
            x=(1-t[:,None,None,None,None])*noise+t[:,None,None,None,None]*target;optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                # Candidate prior is detached exactly at the structure boundary.
                structure_assessment=model(sbatch["reasoning_state"])
                logits=structure_assessment.candidate_logits[:,:candidates.k.shape[1]].masked_fill(~candidates.candidate_mask,-torch.inf)
                structure_prior=torch.softmax(logits,dim=-1).detach()
                output=model(sbatch["reasoning_state"],candidates,x,t,sbatch["atom_mask"])
                breakdown=topology_supervised_candidate_loss(
                    output.velocity,target-noise,x,t,target,sbatch["reasoning_state"].token_mask,
                    sbatch["atom_mask"],candidates,topology_weights)
                structure_loss,_,_=candidate_marginal_loss(breakdown.total,structure_prior,candidates.candidate_mask)
                assessment=model(qbatch["reasoning_state"])
                gate_losses=sequence_gate_objective(assessment,label=qbatch["label"],token_mask=qbatch["reasoning_state"].token_mask,
                    acceptor_index=qbatch["acceptor_index"],plug_index=qbatch["plug_index"],teacher_prior=qbatch["teacher_prior"],
                    candidate_mask=qbatch["candidate_mask"],ood_target=qbatch["ood_target"])
                total=structure_loss+gate_losses["total"]
            total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
            if step%10==0:
                values=torch.tensor([float(total),float(structure_loss),float(gate_losses["total"])],device=device)
                dist.all_reduce(values)
                if rank==0:
                    with (run/"metrics.jsonl").open("a") as handle:handle.write(json.dumps({"event":"metrics","split":"train",
                        "step":step,"loss":float(values[0]/world),"structure_loss":float(values[1]/world),
                        "gate_loss":float(values[2]/world),"time":time.time()})+"\n")
            if step%500==0:_save(model,optimizer,run/f"checkpoint-{step:08d}",step,manifest,rank)
        _save(model,optimizer,run/"checkpoint-final",step,manifest,rank)
    finally:dist.destroy_process_group()


if __name__=="__main__":main()
