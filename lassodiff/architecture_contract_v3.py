"""Fail-closed configuration and checkpoint contract for OpenDDE V3."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ARCHITECTURE_ID_V3 = "lassodiff_opendde_v3"
SCHEMA_VERSION_V3 = 3
REQUIRED_MODULES_V3 = (
    "reasoning_adapter", "sequence_gate", "hypothesis_head", "topology_adapter",
    "structural_tokens", "geometry_encoder", "diffusion",
)


class V3ArchitectureMismatchError(RuntimeError):
    pass


def _required(mapping: Mapping[str, Any], key: str, section: str):
    if key not in mapping:
        raise V3ArchitectureMismatchError(f"V3 config missing {section}.{key}")
    return mapping[key]


@dataclass(frozen=True)
class V3ReasoningConfig:
    backend: str
    checkpoint_path: str
    checkpoint_sha256: str
    opendde_commit: str
    source_root: str
    runtime_root: str
    expected_numel: int
    freeze: bool
    use_msa: bool
    use_template: bool
    cache_dir: str
    cache_schema_version: int
    n_cycle: int
    dtype: str


@dataclass(frozen=True)
class V3ModelConfig:
    c_s: int
    c_z: int
    c_a: int
    n_heads: int
    diffusion_blocks: int
    expected_trainable_numel: int


@dataclass(frozen=True)
class V3SequenceGateConfig:
    enabled: bool
    use_pair_state: bool
    reject_threshold: float
    accept_threshold: float
    target_validation_fpr: float
    ood_threshold: float


@dataclass(frozen=True)
class V3TopologyConfig:
    candidate_specific: bool
    max_candidates: int
    allow_invalid_clamp: bool


@dataclass(frozen=True)
class V3StructureConfig:
    structural_tokens: bool
    dynamic_geometry: bool
    geometry_per_block: bool
    candidate_prior_detach: bool
    reasoning_gradient_scale: float
    target_policy: str
    flow_loss_weight: float
    bond_loss_weight: float
    iso_distance_loss_weight: float
    iso_angle_loss_weight: float
    iso_plane_loss_weight: float
    threading_loss_weight: float


@dataclass(frozen=True)
class V3ScreeningConfig:
    run_diffusion: bool
    hard_iso_projection: bool
    topology_guidance: bool
    allow_user_kp_override: bool


@dataclass(frozen=True)
class V3TrainingConfig:
    opendde_lr: float
    adapter_lr: float
    gate_lr: float
    structure_lr: float


@dataclass(frozen=True)
class V3ArchitectureConfig:
    architecture_id: str
    schema_version: int
    strict: bool
    allow_fallback: bool
    reasoning: V3ReasoningConfig
    model: V3ModelConfig
    sequence_gate: V3SequenceGateConfig
    topology: V3TopologyConfig
    structure: V3StructureConfig
    screening: V3ScreeningConfig
    training: V3TrainingConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "V3ArchitectureConfig":
        def section(name, typ):
            values = _required(raw, name, "root")
            if not isinstance(values, Mapping):
                raise V3ArchitectureMismatchError(f"V3 config section {name} must be a mapping")
            fields = set(typ.__dataclass_fields__)
            missing = fields - set(values)
            if missing:
                raise V3ArchitectureMismatchError(f"V3 config missing {name} fields: {sorted(missing)}")
            return typ(**{field: values[field] for field in fields})

        config = cls(
            architecture_id=_required(raw, "architecture_id", "root"),
            schema_version=int(_required(raw, "schema_version", "root")),
            strict=bool(_required(raw, "strict", "root")),
            allow_fallback=bool(_required(raw, "allow_fallback", "root")),
            reasoning=section("reasoning", V3ReasoningConfig), model=section("model", V3ModelConfig),
            sequence_gate=section("sequence_gate", V3SequenceGateConfig),
            topology=section("topology", V3TopologyConfig), structure=section("structure", V3StructureConfig),
            screening=section("screening", V3ScreeningConfig), training=section("training", V3TrainingConfig),
        )
        config.validate()
        return config

    def validate(self):
        if self.architecture_id != ARCHITECTURE_ID_V3 or self.schema_version != SCHEMA_VERSION_V3:
            raise V3ArchitectureMismatchError("wrong OpenDDE V3 architecture id/schema")
        if not self.strict or self.allow_fallback:
            raise V3ArchitectureMismatchError("V3 must be strict and must not allow fallback")
        r = self.reasoning
        if r.backend != "opendde_pretrained" or not r.freeze or r.use_msa or r.use_template:
            raise V3ArchitectureMismatchError("V3 P1 requires frozen OpenDDE without MSA/template")
        if len(r.checkpoint_sha256) != 64 or len(r.opendde_commit) != 40 or r.expected_numel < 100_000_000:
            raise V3ArchitectureMismatchError("V3 OpenDDE provenance/parameter manifest is invalid")
        if r.cache_schema_version != 1 or r.n_cycle < 1 or r.dtype not in {"float16", "bfloat16", "float32"}:
            raise V3ArchitectureMismatchError("V3 reasoning cache/cycle/dtype contract is invalid")
        m = self.model
        if min(m.c_s, m.c_z, m.c_a, m.n_heads, m.diffusion_blocks, m.expected_trainable_numel) < 1 or m.c_a % m.n_heads:
            raise V3ArchitectureMismatchError("V3 model dimensions are invalid")
        g = self.sequence_gate
        if not g.enabled or not g.use_pair_state or not (0 <= g.reject_threshold < g.accept_threshold <= 1):
            raise V3ArchitectureMismatchError("V3 sequence gate contract is invalid")
        if not (0 < g.target_validation_fpr < 1) or not (0 <= g.ood_threshold <= 1):
            raise V3ArchitectureMismatchError("V3 sequence thresholds are invalid")
        if not self.topology.candidate_specific or self.topology.allow_invalid_clamp:
            raise V3ArchitectureMismatchError("V3 candidates must be specific and invalid candidates must be masked")
        if not 1 <= self.topology.max_candidates <= 3:
            raise V3ArchitectureMismatchError("V3 max_candidates must be between one and three")
        s = self.structure
        if not all((s.structural_tokens, s.dynamic_geometry, s.geometry_per_block, s.candidate_prior_detach)):
            raise V3ArchitectureMismatchError("V3 required structure route is disabled")
        if not 0 < s.reasoning_gradient_scale <= 1:
            raise V3ArchitectureMismatchError("V3 reasoning gradient scale must be in (0,1]")
        if s.target_policy != "topology_valid":
            raise V3ArchitectureMismatchError("V3 structure training requires topology_valid targets")
        loss_weights = (
            s.flow_loss_weight, s.bond_loss_weight, s.iso_distance_loss_weight,
            s.iso_angle_loss_weight, s.iso_plane_loss_weight, s.threading_loss_weight,
        )
        if s.flow_loss_weight <= 0 or any(value < 0 for value in loss_weights):
            raise V3ArchitectureMismatchError("V3 topology loss weights must be non-negative with positive flow")
        if sum(loss_weights[1:]) <= 0:
            raise V3ArchitectureMismatchError("V3 structure training must optimize an explicit topology loss")
        if any(asdict(self.screening).values()):
            raise V3ArchitectureMismatchError("V3 screening cannot generate/project/guide/override")
        if self.training.opendde_lr != 0 or min(
            self.training.adapter_lr, self.training.gate_lr, self.training.structure_lr
        ) <= 0:
            raise V3ArchitectureMismatchError("V3 optimizer learning-rate contract is invalid")

    def expanded(self) -> "V3ArchitectureConfig":
        raw = asdict(self)
        for key in ("checkpoint_path", "source_root", "runtime_root", "cache_dir"):
            raw["reasoning"][key] = os.path.expandvars(raw["reasoning"][key])
        return V3ArchitectureConfig.from_mapping(raw)


def load_architecture_config_v3(path: str | Path) -> V3ArchitectureConfig:
    import yaml
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise V3ArchitectureMismatchError("V3 config root must be a mapping")
    return V3ArchitectureConfig.from_mapping(raw).expanded()


def checkpoint_manifest_v3(config: V3ArchitectureConfig, world_size: int, split_manifest_sha256: str):
    config.validate()
    return {
        "architecture_id": ARCHITECTURE_ID_V3, "schema_version": SCHEMA_VERSION_V3,
        "world_size": int(world_size), "opendde_commit": config.reasoning.opendde_commit,
        "opendde_checkpoint_sha256": config.reasoning.checkpoint_sha256,
        "reasoning_feature_schema_version": config.reasoning.cache_schema_version,
        "reasoner_frozen": True, "required_modules": list(REQUIRED_MODULES_V3),
        "allow_fallback": False, "candidate_specific": True, "screening_projection": False,
        "split_manifest_sha256": str(split_manifest_sha256), "configuration": asdict(config),
    }


def validate_checkpoint_manifest_v3(manifest, config, world_size, split_manifest_sha256):
    expected = checkpoint_manifest_v3(config, world_size, split_manifest_sha256)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise V3ArchitectureMismatchError(f"V3 checkpoint manifest mismatch at {key}")


def sha256_json_file(path: str | Path) -> str:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
