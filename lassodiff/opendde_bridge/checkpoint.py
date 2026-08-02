from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True)
class CheckpointLoadManifest:
    checkpoint_path: str
    checkpoint_sha256: str
    loaded_numel: int
    expected_numel: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_payload(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, ValueError):  # mmap/weights_only support differs by torch release.
        return torch.load(str(path), map_location="cpu")


def _normalized_state(payload) -> Mapping[str, torch.Tensor]:
    state = payload.get("model") if isinstance(payload, Mapping) and "model" in payload else payload
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("OpenDDE checkpoint has no non-empty model state")
    keys = list(state)
    if all(str(key).startswith("module.") for key in keys):
        state = {str(key)[len("module."):]: value for key, value in state.items()}
    return state


def load_checkpoint_strict(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_numel: int | None = None,
) -> CheckpointLoadManifest:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise RuntimeError(f"OpenDDE checkpoint does not exist: {path}")
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"OpenDDE checkpoint SHA-256 mismatch: expected={expected_sha256}, got={actual_sha256}"
        )
    state = _normalized_state(_load_payload(path))
    loaded_numel = sum(int(value.numel()) for value in state.values() if isinstance(value, torch.Tensor))
    model_numel = sum(int(parameter.numel()) for parameter in model.parameters())
    expected = model_numel if expected_numel is None else int(expected_numel)
    if expected_numel is not None and (loaded_numel != expected or model_numel != expected):
        raise RuntimeError(
            f"OpenDDE parameter manifest mismatch: loaded={loaded_numel}, model={model_numel}, expected={expected}"
        )
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"strict OpenDDE checkpoint load failed: {exc}") from exc
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"strict OpenDDE checkpoint load failed: missing={missing}, unexpected={unexpected}")
    if loaded_numel != expected or model_numel != expected:
        raise RuntimeError(
            f"OpenDDE parameter manifest mismatch: loaded={loaded_numel}, model={model_numel}, expected={expected}"
        )
    return CheckpointLoadManifest(
        checkpoint_path=str(path.resolve()), checkpoint_sha256=actual_sha256,
        loaded_numel=loaded_numel, expected_numel=expected,
        missing_keys=missing, unexpected_keys=unexpected,
    )
