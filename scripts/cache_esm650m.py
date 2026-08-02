#!/usr/bin/env python3
"""Precompute frozen ESM-2 650M residue embeddings into a versioned LMDB."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lassodiff.data.lassopred_lmdb import LassoPredLMDBDataset, write_esm_cache

MODEL_NAME = "esm2_t33_650M_UR50D"
REPRESENTATION_LAYER = 33


def batches(sequences, batch_size):
    for start in range(0, len(sequences), batch_size):
        yield sequences[start:start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="LassoPred coordinate LMDB")
    parser.add_argument("--output", required=True, help="ESM cache LMDB")
    parser.add_argument("--batch-size", type=int, default=1, help="650M-safe default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--map-size-gb", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was supplied but CUDA is not available")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("warning: generating ESM-2 650M cache on CPU will be very slow")
    try:
        import esm
    except ImportError as exc:
        raise RuntimeError("fair-esm is required to create the ESM cache") from exc
    dataset = LassoPredLMDBDataset(args.dataset)
    sequences = list(OrderedDict.fromkeys(dataset[index]["sequence"] for index in range(len(dataset))))
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    converter = alphabet.get_batch_converter()
    embeddings = []
    with torch.inference_mode():
        for index, batch in enumerate(batches(sequences, args.batch_size), start=1):
            _, _, tokens = converter([(str(i), sequence) for i, sequence in enumerate(batch)])
            tokens = tokens.to(device)
            autocast = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.no_grad()
            with autocast:
                output = model(tokens, repr_layers=[REPRESENTATION_LAYER], return_contacts=False)
            representations = output["representations"][REPRESENTATION_LAYER]
            for row, sequence in enumerate(batch):
                embeddings.append((sequence, representations[row, 1:len(sequence) + 1].cpu()))
            print(f"cached {min(index * args.batch_size, len(sequences))}/{len(sequences)} sequences", flush=True)
    manifest = write_esm_cache(
        args.output, MODEL_NAME, REPRESENTATION_LAYER, embeddings,
        map_size=args.map_size_gb * 1024**3, overwrite=args.overwrite,
    )
    print(manifest)


if __name__ == "__main__":
    main()
