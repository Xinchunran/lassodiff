#!/usr/bin/env python3
"""Fail-closed completeness/roundtrip verifier for the full reasoning cache."""
from __future__ import annotations

import argparse,hashlib,json,pickle,sys
from pathlib import Path

_ROOT=Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:sys.path.insert(0,str(_ROOT))

from lassodiff.architecture_contract_v3 import load_architecture_config_v3
from lassodiff.data.lassopred_lmdb import LassoPredLMDBDataset,_RECORD_PREFIX
from lassodiff.opendde_bridge.cache import OpenDDEReasoningCache,ReasoningCacheIdentity


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    parser.add_argument("--dataset",required=True);parser.add_argument("--output",required=True);args=parser.parse_args()
    config=load_architecture_config_v3(args.config);dataset=LassoPredLMDBDataset(args.dataset);sequences=set()
    with dataset._open().begin() as txn:
        for record_id in dataset.record_ids:
            sequences.add(pickle.loads(txn.get(_RECORD_PREFIX+record_id.encode()))["sequence"])
    cache=OpenDDEReasoningCache(config.reasoning.cache_dir);single_elements=pair_elements=0;entries=[]
    for sequence in sorted(sequences):
        identity=ReasoningCacheIdentity(sequence,config.reasoning.opendde_commit,config.reasoning.checkpoint_sha256,
            config.reasoning.cache_schema_version,False,False,config.reasoning.dtype,config.reasoning.n_cycle)
        state=cache.get(identity)
        if state.single.shape[1]!=len(sequence) or state.pair.shape[1:3]!=(len(sequence),len(sequence)):
            raise RuntimeError("cache sequence length/state shape mismatch")
        single_elements+=state.single.numel();pair_elements+=state.pair.numel();entries.append(cache.key(identity))
    entries_sha=hashlib.sha256("\n".join(entries).encode()).hexdigest()
    report={"status":"PASS","schema_version":1,"sequence_count":len(sequences),
        "dataset_record_count":len(dataset),"cache_dir":str(Path(config.reasoning.cache_dir).resolve()),
        "opendde_commit":config.reasoning.opendde_commit,"checkpoint_sha256":config.reasoning.checkpoint_sha256,
        "feature_schema_version":config.reasoning.cache_schema_version,"n_cycle":config.reasoning.n_cycle,
        "dtype":config.reasoning.dtype,"single_elements":single_elements,"pair_elements":pair_elements,
        "ordered_cache_keys_sha256":entries_sha}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,sort_keys=True))


if __name__=="__main__":main()
