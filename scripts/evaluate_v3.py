#!/usr/bin/env python3
"""Locked 4-GPU rollout evaluator for a sharded V3 structure checkpoint."""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardedStateDictConfig, StateDictType
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_ROOT=Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:sys.path.insert(0,str(_ROOT))

from lassodiff.architecture_contract_v3 import load_architecture_config_v3, sha256_json_file, validate_checkpoint_manifest_v3
from lassodiff.data.lassopred_lmdb import LassoPredLMDBDataset
from lassodiff.data.family_split import validate_family_split_manifest
from lassodiff.data.opendde_cached import OpenDDECachedDataset, collate_lassodiff_v3
from lassodiff.model_v3 import LassoDiffOpenDDEV3
from lassodiff.opendde_bridge.cache import OpenDDEReasoningCache
from lassodiff.opendde_bridge.reasoner import CacheOnlyOpenDDEReasoner
from scripts.train_structure_v3 import _identity_builder, _rollout_validation, _apply_diffusion_checkpointing


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    parser.add_argument("--dataset",required=True);parser.add_argument("--locked-split",required=True)
    parser.add_argument("--checkpoint",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--batch-size",type=int,default=4);parser.add_argument("--sampling-steps",type=int,default=40)
    parser.add_argument("--num-workers",type=int,default=2);args=parser.parse_args()
    rank,world,local=int(os.environ["RANK"]),int(os.environ["WORLD_SIZE"]),int(os.environ["LOCAL_RANK"])
    if world!=4 or not torch.cuda.is_available():raise RuntimeError("locked V3 evaluation requires four CUDA ranks")
    torch.cuda.set_device(local);device=torch.device("cuda",local);dist.init_process_group("nccl",device_id=device)
    try:
        config=load_architecture_config_v3(args.config);split=json.loads(Path(args.locked_split).read_text())
        validate_family_split_manifest(split)
        split_sha=sha256_json_file(args.locked_split)
        base=LassoPredLMDBDataset(args.dataset,split["test"],seed=1701)
        dataset=OpenDDECachedDataset(base,OpenDDEReasoningCache(config.reasoning.cache_dir),_identity_builder(config))
        sampler=DistributedSampler(dataset,world,rank,shuffle=False,drop_last=False)
        loader=DataLoader(dataset,batch_size=args.batch_size,sampler=sampler,num_workers=args.num_workers,
                          collate_fn=collate_lassodiff_v3,drop_last=False,pin_memory=True)
        module=LassoDiffOpenDDEV3(CacheOnlyOpenDDEReasoner(config.reasoning.checkpoint_sha256,config.reasoning.opendde_commit),
            384,384,c_s=config.model.c_s,c_z=config.model.c_z,c_a=config.model.c_a,n_heads=config.model.n_heads,
            diffusion_blocks=config.model.diffusion_blocks,max_candidates=config.topology.max_candidates,
            structure_gradient_scale=config.structure.reasoning_gradient_scale).to(device)
        _apply_diffusion_checkpointing(module)
        model=FSDP(module,device_id=device,use_orig_params=True,mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.bfloat16))
        checkpoint=torch.load(Path(args.checkpoint)/f"checkpoint.rank{rank:04d}.pt",map_location="cpu",weights_only=False)
        validate_checkpoint_manifest_v3(checkpoint["architecture_manifest"],config,world,split_sha)
        with FSDP.state_dict_type(model,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=True)):
            model.load_state_dict(checkpoint["model"],strict=True)
        metrics=_rollout_validation(model,loader,device,args.sampling_steps,max_batches=len(loader))
        if rank==0:
            report={"architecture_id":config.architecture_id,"schema_version":3,"checkpoint_step":checkpoint["step"],
                "opendde_checkpoint_sha256":config.reasoning.checkpoint_sha256,"opendde_commit":config.reasoning.opendde_commit,
                "split_manifest_sha256":split_sha,"projection_used":False,"topology_guidance_used":False,
                "metrics_schema_version":3,**metrics}
            output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,sort_keys=True))
    finally:dist.destroy_process_group()


if __name__=="__main__":main()
