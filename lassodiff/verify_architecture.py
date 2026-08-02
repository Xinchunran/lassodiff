"""Fail-closed preflight verifier for the OpenDDE V3 development line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from .topology_adapter import CandidateBatch


def _main_v3(args) -> None:
    from .architecture_contract_v3 import load_architecture_config_v3
    from .model_v3 import LassoDiffOpenDDEV3
    from .opendde_bridge.feature_builder import OpenDDEFeatureBuilder
    from .opendde_bridge.loader import PinnedOpenDDEConfig, load_pinned_opendde_reasoner
    from .preflight_v3 import build_v3_optimizer, verify_model_contract_v3

    config = load_architecture_config_v3(args.config)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V3 preflight requested CUDA but CUDA is unavailable")
        torch.cuda.set_device(device.index or 0)
    reasoner, load_manifest = load_pinned_opendde_reasoner(PinnedOpenDDEConfig(
        source_root=config.reasoning.source_root, runtime_root=config.reasoning.runtime_root,
        checkpoint_path=config.reasoning.checkpoint_path,
        checkpoint_sha256=config.reasoning.checkpoint_sha256,
        opendde_commit=config.reasoning.opendde_commit,
        expected_numel=config.reasoning.expected_numel, n_cycle=config.reasoning.n_cycle,
        device="cuda" if device.type == "cuda" else "cpu", dtype=config.reasoning.dtype,
    ))
    builder = OpenDDEFeatureBuilder(config.reasoning.runtime_root)
    sequence = "MDELAKILGATPEEIRKALENAGADVVVVDSVAALTPA"
    features = builder.build(sequence, name="lassodiff_v3_preflight")
    model = LassoDiffOpenDDEV3(
        reasoner, int(reasoner.model.c_s), int(reasoner.model.c_z),
        c_s=config.model.c_s, c_z=config.model.c_z, c_a=config.model.c_a,
        n_heads=config.model.n_heads, diffusion_blocks=config.model.diffusion_blocks,
        max_candidates=config.topology.max_candidates,
        structure_gradient_scale=config.structure.reasoning_gradient_scale,
    ).to(device)
    candidates = CandidateBatch(
        k=torch.tensor([[2, 2]], device=device), p=torch.tensor([[7, 9]], device=device),
        acceptor_index=torch.tensor([[1, 1]], device=device),
        prior=torch.tensor([[0.7, 0.3]], device=device),
        candidate_mask=torch.tensor([[True, True]], device=device),
    )
    optimizer = build_v3_optimizer(model, config)
    result = verify_model_contract_v3(
        model, None, candidates, optimizer, config, reasoner_features=features,
    )
    result["checkpoint_load_manifest"] = load_manifest.__dict__
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    try:
        _main_v3(args)
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
