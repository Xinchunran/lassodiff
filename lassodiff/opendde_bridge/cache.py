from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

from .schema import OpenDDEReasoningState, validate_reasoning_state
from ..seq_encoder import seq_to_aa_ids


@dataclass(frozen=True)
class ReasoningCacheIdentity:
    sequence: str
    opendde_commit: str
    checkpoint_sha256: str
    feature_schema_version: int
    use_msa: bool
    use_template: bool
    dtype: str
    n_cycle: int = 10

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest()

    def manifest(self):
        return {**asdict(self), "sequence_sha256": self.sequence_sha256}


class OpenDDEReasoningCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    @staticmethod
    def key(identity: ReasoningCacheIdentity) -> str:
        payload = json.dumps(identity.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def path(self, identity: ReasoningCacheIdentity) -> Path:
        return self.root / self.key(identity)

    def put(
        self, identity: ReasoningCacheIdentity, state: OpenDDEReasoningState, *, overwrite: bool = False,
    ) -> Path:
        state = validate_reasoning_state(state)
        if state.sequence_hashes != (identity.sequence_sha256,):
            raise RuntimeError("reasoning state sequence does not match cache identity")
        expected_provenance = (identity.opendde_commit, identity.checkpoint_sha256, identity.feature_schema_version)
        got_provenance = (state.opendde_commit, state.checkpoint_sha256, state.feature_schema_version)
        if got_provenance != expected_provenance:
            raise RuntimeError("reasoning state provenance does not match cache identity")
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.path(identity)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite OpenDDE reasoning cache entry: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(self.root)))
        try:
            cache_dtype = {
                "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32,
            }.get(identity.dtype)
            if cache_dtype is None:
                raise RuntimeError(f"unsupported OpenDDE reasoning cache dtype: {identity.dtype}")
            tensors = {
                "single": state.single[0].detach().cpu().to(cache_dtype).contiguous(),
                "pair": state.pair[0].detach().cpu().to(cache_dtype).contiguous(),
                "token_mask": state.token_mask[0].detach().cpu().to(torch.uint8).contiguous(),
                "residue_index": state.residue_index[0].detach().cpu().long().contiguous(),
            }
            save_file(tensors, str(temporary / "state.safetensors"))
            metadata = {
                **identity.manifest(),
                "single_shape": list(tensors["single"].shape),
                "pair_shape": list(tensors["pair"].shape),
            }
            (temporary / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            if destination.exists():
                shutil.rmtree(destination)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return destination

    def get(self, identity: ReasoningCacheIdentity) -> OpenDDEReasoningState:
        directory = self.path(identity)
        try:
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            tensors = load_file(str(directory / "state.safetensors"), device="cpu")
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenDDE reasoning cache is missing or corrupt: {directory}") from exc
        expected = identity.manifest()
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise RuntimeError("OpenDDE reasoning cache identity mismatch")
        state = OpenDDEReasoningState(
            single=tensors["single"][None], pair=tensors["pair"][None],
            token_mask=tensors["token_mask"].bool()[None], residue_index=tensors["residue_index"][None],
            sequence_hashes=(identity.sequence_sha256,), checkpoint_sha256=identity.checkpoint_sha256,
            opendde_commit=identity.opendde_commit, feature_schema_version=identity.feature_schema_version,
            residue_type=seq_to_aa_ids(identity.sequence)[None],
        )
        if list(state.single.shape[1:]) != metadata.get("single_shape") or list(state.pair.shape[1:]) != metadata.get("pair_shape"):
            raise RuntimeError("OpenDDE reasoning cache tensor shape mismatch")
        return validate_reasoning_state(state)
