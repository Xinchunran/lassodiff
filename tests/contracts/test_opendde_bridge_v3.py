from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lassodiff.opendde_bridge.checkpoint import load_checkpoint_strict, sha256_file
from lassodiff.opendde_bridge.reasoner import FrozenOpenDDEReasoner
from lassodiff.opendde_bridge.schema import OpenDDEReasoningState, validate_reasoning_state


class TinyReasoningModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.single = torch.nn.Linear(4, 6)
        self.pair = torch.nn.Linear(4, 5)

    def forward_reasoning(self, features):
        x = features["x"]
        single = self.single(x)
        pair_input = x[:, :, None] + x[:, None, :]
        return self.pair(pair_input), single


def _checkpoint(path: Path, model: torch.nn.Module, remove_key: str | None = None):
    state = model.state_dict()
    if remove_key:
        state.pop(remove_key)
    torch.save({"model": {f"module.{key}": value for key, value in state.items()}}, path)


def test_missing_checkpoint_is_fatal(tmp_path: Path):
    with pytest.raises(RuntimeError, match="OpenDDE checkpoint"):
        load_checkpoint_strict(TinyReasoningModel(), tmp_path / "missing.pt")


def test_checkpoint_load_is_strict(tmp_path: Path):
    path = tmp_path / "broken.pt"
    _checkpoint(path, TinyReasoningModel(), remove_key="single.bias")
    with pytest.raises(RuntimeError, match="strict OpenDDE checkpoint load failed"):
        load_checkpoint_strict(TinyReasoningModel(), path)


def test_loaded_parameter_manifest_matches(tmp_path: Path):
    source = TinyReasoningModel()
    path = tmp_path / "ok.pt"
    _checkpoint(path, source)
    expected_numel = sum(parameter.numel() for parameter in source.parameters())
    manifest = load_checkpoint_strict(
        TinyReasoningModel(), path, expected_sha256=sha256_file(path), expected_numel=expected_numel
    )
    assert manifest.loaded_numel == manifest.expected_numel == expected_numel
    assert manifest.missing_keys == ()
    assert manifest.unexpected_keys == ()


def test_frozen_reasoner_stays_eval_after_train():
    model = TinyReasoningModel()

    def forward(features):
        pair, single = model.forward_reasoning(features)
        B, L = single.shape[:2]
        return OpenDDEReasoningState(
            single=single, pair=pair, token_mask=torch.ones(B, L, dtype=torch.bool),
            residue_index=torch.arange(L)[None].expand(B, -1),
            sequence_hashes=("a",) * B, checkpoint_sha256="hash", opendde_commit="commit",
            feature_schema_version=1,
        )

    reasoner = FrozenOpenDDEReasoner(model, forward)
    reasoner.train()
    assert reasoner.training
    assert reasoner.model.training is False
    assert all(not parameter.requires_grad for parameter in reasoner.model.parameters())
    assert validate_reasoning_state(reasoner({"x": torch.randn(1, 3, 4)})).single.shape == (1, 3, 6)


def test_reasoning_state_requires_provenance():
    state = OpenDDEReasoningState(
        single=torch.zeros(1, 2, 3), pair=torch.zeros(1, 2, 2, 4),
        token_mask=torch.ones(1, 2, dtype=torch.bool), residue_index=torch.arange(2)[None],
        sequence_hashes=("abc",), checkpoint_sha256="", opendde_commit="commit",
        feature_schema_version=1,
    )
    with pytest.raises(ValueError, match="provenance"):
        validate_reasoning_state(state)
