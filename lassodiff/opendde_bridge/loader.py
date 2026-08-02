from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess

from .checkpoint import CheckpointLoadManifest, sha256_file
from .reasoner import FrozenOpenDDEReasoner, PinnedOpenDDEForward


@dataclass(frozen=True)
class PinnedOpenDDEConfig:
    source_root: str
    runtime_root: str
    checkpoint_path: str
    checkpoint_sha256: str
    opendde_commit: str
    expected_numel: int
    n_cycle: int = 10
    device: str = "cpu"
    dtype: str = "fp32"


def _verify_source(config: PinnedOpenDDEConfig):
    source = Path(config.source_root)
    if not (source / "opendde" / "model" / "opendde.py").is_file():
        raise RuntimeError(f"pinned OpenDDE source is missing: {source}")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot verify pinned OpenDDE source commit") from exc
    if commit != config.opendde_commit:
        raise RuntimeError(f"OpenDDE source commit mismatch: expected={config.opendde_commit}, got={commit}")


def load_pinned_opendde_reasoner(config: PinnedOpenDDEConfig):
    _verify_source(config)
    checkpoint = Path(config.checkpoint_path)
    if not checkpoint.is_file():
        raise RuntimeError(f"OpenDDE checkpoint does not exist: {checkpoint}")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != config.checkpoint_sha256:
        raise RuntimeError(
            f"OpenDDE checkpoint SHA-256 mismatch: expected={config.checkpoint_sha256}, got={actual_sha}"
        )
    os.environ["OPENDDE_ROOT_DIR"] = str(Path(config.runtime_root).resolve())
    from runner.batch_inference import get_default_runner

    inference_dtype = "bf16" if config.dtype in {"float16", "bfloat16"} and str(config.device).startswith("cuda") else "fp32"
    runner = get_default_runner(
        seeds=[101], dump_dir=str(Path(config.runtime_root) / "lassodiff_bridge_output"),
        n_cycle=config.n_cycle, n_step=2, n_sample=1, dtype=inference_dtype,
        model_name="opendde_v1", load_checkpoint_path=str(checkpoint),
        use_msa=False, use_template=False, use_rna_msa=False,
        trimul_kernel="torch", triatt_kernel="torch", enable_cache=False,
        enable_fusion=False, enable_tf32=False, deterministic=True,
        need_atom_confidence=False, use_tfg_guidance=False, device=config.device,
    )
    if not bool(runner.configs.load_strict):
        raise RuntimeError("OpenDDE runner did not enforce strict checkpoint loading")
    model_numel = sum(parameter.numel() for parameter in runner.model.parameters())
    if model_numel != int(config.expected_numel):
        raise RuntimeError(
            f"OpenDDE model parameter manifest mismatch: expected={config.expected_numel}, got={model_numel}"
        )
    forward = PinnedOpenDDEForward(
        runner.model, checkpoint_sha256=actual_sha, opendde_commit=config.opendde_commit,
        n_cycle=config.n_cycle, inference_dtype=inference_dtype,
    )
    reasoner = FrozenOpenDDEReasoner(runner.model, forward)
    reasoner._opendde_runner = runner  # Keep runner/runtime ownership alive.
    manifest = CheckpointLoadManifest(
        checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=actual_sha,
        loaded_numel=model_numel, expected_numel=int(config.expected_numel),
        missing_keys=(), unexpected_keys=(),
    )
    return reasoner, manifest
