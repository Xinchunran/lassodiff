from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from lassodiff.architecture_contract_v3 import V3ArchitectureConfig
from lassodiff.model_v3 import LassoDiffOpenDDEV3
from lassodiff.opendde_bridge.reasoner import FrozenOpenDDEReasoner
from lassodiff.opendde_bridge.schema import OpenDDEReasoningState
from lassodiff.preflight_v3 import build_v3_optimizer, verify_model_contract_v3
from lassodiff.topology_adapter import CandidateBatch


def _config():
    return V3ArchitectureConfig.from_mapping({
        "architecture_id": "lassodiff_opendde_v3", "schema_version": 3,
        "strict": True, "allow_fallback": False,
        "reasoning": {"backend": "opendde_pretrained", "checkpoint_path": "/x", "checkpoint_sha256": "h" * 64,
            "opendde_commit": "c" * 40, "source_root": "/s", "runtime_root": "/r", "expected_numel": 655791538,
            "freeze": True, "use_msa": False, "use_template": False, "cache_dir": "/c",
            "cache_schema_version": 1, "n_cycle": 10, "dtype": "float16"},
        "model": {"c_s": 16, "c_z": 12, "c_a": 24, "n_heads": 4, "diffusion_blocks": 2,
                  "expected_trainable_numel": 20631},
        "sequence_gate": {"enabled": True, "use_pair_state": True, "reject_threshold": .1,
            "accept_threshold": .8, "target_validation_fpr": .01, "ood_threshold": .9},
        "topology": {"candidate_specific": True, "max_candidates": 3, "allow_invalid_clamp": False},
        "structure": {"structural_tokens": True, "dynamic_geometry": True, "geometry_per_block": True,
            "candidate_prior_detach": True, "reasoning_gradient_scale": .1,
            "target_policy": "topology_valid", "flow_loss_weight": 1.0,
            "bond_loss_weight": 1.0, "iso_distance_loss_weight": 10.0,
            "iso_angle_loss_weight": 2.0, "iso_plane_loss_weight": 2.0,
            "threading_loss_weight": 2.0},
        "screening": {"run_diffusion": False, "hard_iso_projection": False,
            "topology_guidance": False, "allow_user_kp_override": False},
        "training": {"opendde_lr": 0., "adapter_lr": 1e-4, "gate_lr": 1e-4, "structure_lr": 1e-4},
    })


def _state():
    torch.manual_seed(101)
    return OpenDDEReasoningState(
        torch.randn(1, 6, 9), torch.randn(1, 6, 6, 7), torch.ones(1, 6, dtype=torch.bool),
        torch.arange(6)[None], ("seq",), "h" * 64, "c" * 40, 1,
    )


def _model():
    core = torch.nn.Linear(1, 1)
    reasoner = FrozenOpenDDEReasoner(core, lambda _: _state())
    return LassoDiffOpenDDEV3(
        reasoner, 9, 7, c_s=16, c_z=12, c_a=24, n_heads=4, diffusion_blocks=2,
    )


def _candidates():
    return CandidateBatch(
        torch.tensor([[1, 1]]), torch.tensor([[3, 4]]), torch.tensor([[2, 2]]),
        torch.tensor([[.7, .3]]), torch.tensor([[True, True]]),
    )


def test_preflight_proves_route_gradient_optimizer_and_screening_contract():
    model, config = _model(), _config()
    optimizer = build_v3_optimizer(model, config)
    result = verify_model_contract_v3(model, _state(), _candidates(), optimizer, config, reasoner_features={})
    assert result["status"] == "PASS"
    assert result["trace"]["geometry_calls"] == 2
    assert result["trace"]["reasoner_called"] == 1
    assert result["trace"]["sequence_gate_called"] == 1
    assert result["trace"]["reasoning_adapter_called"] >= 2
    assert result["sequence_pair_relative_change"] > 1e-6
    assert result["structure_pair_relative_change"] > 1e-6


def test_preflight_rejects_an_optimizer_missing_required_parameter():
    model, config = _model(), _config()
    optimizer = torch.optim.AdamW(model.sequence_gate.parameters(), lr=1e-4)
    with pytest.raises(RuntimeError, match="optimizer missing"):
        verify_model_contract_v3(model, _state(), _candidates(), optimizer, config, reasoner_features={})
