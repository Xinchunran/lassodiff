#!/usr/bin/env python3
"""Build a versioned verified-positive/multi-stratum negative gate manifest."""
from __future__ import annotations

import argparse,hashlib,json,random,sys
from pathlib import Path

_ROOT=Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:sys.path.insert(0,str(_ROOT))

from lassodiff.data.sequence_v3 import SequenceExample,build_sequence_split,validate_sequence_examples


def _fasta(path):
    result=[];name=None;parts=[]
    for line in Path(path).read_text().splitlines():
        line=line.strip()
        if not line:continue
        if line.startswith(">"):
            if name is not None:result.append((name,"".join(parts).upper()))
            name=line[1:].split()[0];parts=[]
        else:parts.append(line)
    if name is not None:result.append((name,"".join(parts).upper()))
    return result


def _shuffle(sequence,rng):
    values=list(sequence)
    for _ in range(20):
        rng.shuffle(values);candidate="".join(values)
        if candidate!=sequence:return candidate
    return sequence[::-1]


def _mutate(sequence,index,residue):
    return sequence[:index]+residue+sequence[index+1:]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--verified-positives",required=True,
        help="JSON list with id, sequence, family_id, acceptor_index, plug_index, optional teacher_prior")
    parser.add_argument("--background-fasta",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,default=1701)
    parser.add_argument("--max-sequence-distance",type=float,default=.4)
    parser.add_argument("--min-stratum-size",type=int,default=5)
    args=parser.parse_args();rng=random.Random(args.seed)
    positives=json.loads(Path(args.verified_positives).read_text());examples=[];source_groups={}
    for row in positives:
        evidence=row.get("evidence")
        if evidence not in {"verified_lasso","experimental_lasso"}:raise RuntimeError("positive evidence is not verified")
        source=str(row["id"]);sequence=str(row["sequence"]).upper();group=str(row["family_id"])
        acceptor=int(row["acceptor_index"]);plug=int(row["plug_index"])
        examples.append(SequenceExample(source,sequence,1,"positive",evidence,group,acceptor,plug,
                                        teacher_prior=row.get("teacher_prior")))
        derived=[
            (f"{source}:shuffle",_shuffle(sequence,rng),"composition_shuffle","composition_preserving_shuffle"),
            (f"{source}:acceptor_mutant",_mutate(sequence,acceptor,"N" if sequence[acceptor]=="D" else "Q"),
             "hard_mutant","acceptor_hard_mutant"),
            (f"{source}:wrong_topology",_mutate(sequence,plug,"A" if sequence[plug]!="A" else "G"),
             "wrong_topology","plug_wrong_topology_mutant"),
        ]
        for identifier,negative,kind,negative_evidence in derived:
            examples.append(SequenceExample(identifier,negative,0,kind,negative_evidence,group,ood=False))
            source_groups[identifier]=group
    background=_fasta(args.background_fasta)
    for identifier,sequence in background:
        group="background:"+hashlib.sha256(sequence.encode()).hexdigest()[:16]
        examples.append(SequenceExample(f"background:{identifier}",sequence,0,"background","background_non_lasso",group))
    random_count=max(1,round(len(examples)*.1))
    alphabet="ACDEFGHIKLMNPQRSTVWY"
    lengths=[len(row.sequence) for row in examples]
    for index in range(random_count):
        length=rng.choice(lengths);sequence="".join(rng.choice(alphabet) for _ in range(length))
        examples.append(SequenceExample(f"random:{index}",sequence,0,"random","random_ood",f"random:{index}",ood=True))
    validate_sequence_examples(examples,source_groups=source_groups)
    split_contract=build_sequence_split(
        examples,seed=args.seed,max_sequence_distance=args.max_sequence_distance,
        min_stratum_size=args.min_stratum_size,
    )
    rows=[]
    for row in examples:
        payload=row.__dict__.copy()
        if isinstance(payload.get("teacher_prior"),tuple):payload["teacher_prior"]=list(payload["teacher_prior"])
        rows.append(payload)
    manifest={"schema_version":2,"seed":args.seed,"examples":rows,"source_groups":source_groups,
              "split":split_contract["split"],"split_contract":split_contract,
              "negative_strata":sorted({row.kind for row in examples if row.label==0}),
              "positive_evidence_required":True}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print(json.dumps({
        "examples":len(examples),"positives":len(positives),"negative_strata":manifest["negative_strata"]},sort_keys=True))


if __name__=="__main__":main()
