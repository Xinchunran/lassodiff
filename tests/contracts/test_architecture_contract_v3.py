from __future__ import annotations

import copy

import pytest

from lassodiff.architecture_contract_v3 import (
    V3ArchitectureMismatchError,
    V3ArchitectureConfig,
    checkpoint_manifest_v3,
    validate_checkpoint_manifest_v3,
)


def _mapping():
    return {
        "architecture_id": "lassodiff_opendde_v3",
        "schema_version": 3,
        "strict": True,
        "allow_fallback": False,
        "reasoning": {
            "backend": "opendde_pretrained", "checkpoint_path": "/checkpoint.pt",
            "checkpoint_sha256": "a" * 64, "opendde_commit": "b" * 40,
            "source_root": "/source", "runtime_root": "/runtime", "expected_numel": 655791538,
            "freeze": True, "use_msa": False, "use_template": False,
            "cache_dir": "/cache", "cache_schema_version": 1, "n_cycle": 10, "dtype": "float16",
        },
        "model": {"c_s": 384, "c_z": 192, "c_a": 384, "n_heads": 8, "diffusion_blocks": 16,
                  "expected_trainable_numel": 29948119},
        "sequence_gate": {
            "enabled": True, "use_pair_state": True, "reject_threshold": 0.10,
            "accept_threshold": 0.80, "target_validation_fpr": 0.01, "ood_threshold": 0.90,
        },
        "topology": {"candidate_specific": True, "max_candidates": 3, "allow_invalid_clamp": False},
        "structure": {
            "structural_tokens": True, "dynamic_geometry": True, "geometry_per_block": True,
            "candidate_prior_detach": True, "reasoning_gradient_scale": 0.1,
            "target_policy": "topology_valid", "flow_loss_weight": 1.0,
            "bond_loss_weight": 1.0, "iso_distance_loss_weight": 10.0,
            "iso_angle_loss_weight": 2.0, "iso_plane_loss_weight": 2.0,
            "threading_loss_weight": 2.0,
        },
        "screening": {
            "run_diffusion": False, "hard_iso_projection": False,
            "topology_guidance": False, "allow_user_kp_override": False,
        },
        "training": {"opendde_lr": 0.0, "adapter_lr": 1e-4, "gate_lr": 1e-4, "structure_lr": 1e-4},
    }


def test_v3_contract_rejects_fallback_and_unfrozen_reasoner():
    for path, value in (("allow_fallback", True), ("reasoning.freeze", False)):
        raw = copy.deepcopy(_mapping())
        if "." in path:
            first, second = path.split(".")
            raw[first][second] = value
        else:
            raw[path] = value
        with pytest.raises(V3ArchitectureMismatchError):
            V3ArchitectureConfig.from_mapping(raw)


def test_v3_contract_rejects_screening_generation_or_guidance():
    for key in ("run_diffusion", "hard_iso_projection", "topology_guidance", "allow_user_kp_override"):
        raw = copy.deepcopy(_mapping())
        raw["screening"][key] = True
        with pytest.raises(V3ArchitectureMismatchError):
            V3ArchitectureConfig.from_mapping(raw)


def test_v3_contract_rejects_velocity_only_structure_training():
    raw = copy.deepcopy(_mapping())
    for key in (
        "bond_loss_weight", "iso_distance_loss_weight", "iso_angle_loss_weight",
        "iso_plane_loss_weight", "threading_loss_weight",
    ):
        raw["structure"][key] = 0.0
    with pytest.raises(V3ArchitectureMismatchError, match="explicit topology loss"):
        V3ArchitectureConfig.from_mapping(raw)


def test_v3_checkpoint_manifest_is_exact_and_rejects_v2():
    config = V3ArchitectureConfig.from_mapping(_mapping())
    manifest = checkpoint_manifest_v3(config, world_size=4, split_manifest_sha256="c" * 64)
    validate_checkpoint_manifest_v3(manifest, config, world_size=4, split_manifest_sha256="c" * 64)
    wrong = copy.deepcopy(manifest)
    wrong["schema_version"] = 2
    with pytest.raises(V3ArchitectureMismatchError):
        validate_checkpoint_manifest_v3(wrong, config, world_size=4, split_manifest_sha256="c" * 64)


def test_checkpoint_manifest_lists_every_required_module():
    config = V3ArchitectureConfig.from_mapping(_mapping())
    manifest = checkpoint_manifest_v3(config, world_size=4, split_manifest_sha256="c" * 64)
    assert set(manifest["required_modules"]) == {
        "reasoning_adapter", "sequence_gate", "hypothesis_head", "topology_adapter",
        "structural_tokens", "geometry_encoder", "diffusion",
    }
    assert manifest["reasoner_frozen"] is True
    assert manifest["candidate_specific"] is True
    assert manifest["allow_fallback"] is False
