#!/usr/bin/env python3
"""Cache residue embeddings without target/candidate/fold information."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lassodiff.esm_encoder_mini import FrozenESMResidueEncoder, embedding_cache_key, save_embedding_cache_item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True, help="newline-delimited normalized sequences")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder-name", default="esm2_t30_150M_UR50D")
    parser.add_argument("--encoder-revision", default="main")
    args = parser.parse_args()
    sequences = sorted({"".join(line.upper().split()) for line in Path(args.sequences).read_text().splitlines() if line.strip()})
    encoder = FrozenESMResidueEncoder(args.encoder_name, args.encoder_revision)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest = {"encoder_name": args.encoder_name, "encoder_revision": args.encoder_revision, "items": []}
    for sequence in sequences:
        import torch
        mask = torch.ones((1, len(sequence)), dtype=torch.bool)
        embedding = encoder([sequence], mask)[0]
        key = embedding_cache_key(args.encoder_name, args.encoder_revision, sequence)
        save_embedding_cache_item(output / f"{key}.pt", sequence, embedding, args.encoder_name, args.encoder_revision)
        manifest["items"].append({"sequence": sequence, "sha256": key, "path": f"{key}.pt"})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
