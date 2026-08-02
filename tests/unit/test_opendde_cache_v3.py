from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from lassodiff.opendde_bridge.cache import OpenDDEReasoningCache, ReasoningCacheIdentity
from lassodiff.opendde_bridge.schema import OpenDDEReasoningState


def _identity(sequence="ACDE", checkpoint="checkpoint", schema=1):
    return ReasoningCacheIdentity(
        sequence=sequence, opendde_commit="commit", checkpoint_sha256=checkpoint,
        feature_schema_version=schema, use_msa=False, use_template=False, dtype="float16", n_cycle=10,
    )


def _state(identity):
    L = len(identity.sequence)
    return OpenDDEReasoningState(
        single=torch.randn(1, L, 6), pair=torch.randn(1, L, L, 5),
        token_mask=torch.ones(1, L, dtype=torch.bool), residue_index=torch.arange(L)[None],
        sequence_hashes=(identity.sequence_sha256,), checkpoint_sha256=identity.checkpoint_sha256,
        opendde_commit=identity.opendde_commit, feature_schema_version=identity.feature_schema_version,
    )


def test_cache_key_changes_with_sequence_checkpoint_and_schema(tmp_path):
    cache = OpenDDEReasoningCache(tmp_path)
    base = cache.key(_identity())
    assert base != cache.key(_identity(sequence="ACDF"))
    assert base != cache.key(_identity(checkpoint="other"))
    assert base != cache.key(_identity(schema=2))


def test_cache_roundtrip_and_padding_mask(tmp_path):
    cache = OpenDDEReasoningCache(tmp_path)
    identity = _identity()
    state = _state(identity)
    state.token_mask[:, -1] = False
    state.single[:, -1] = 0
    state.pair[:, -1] = 0
    state.pair[:, :, -1] = 0
    cache.put(identity, state)
    loaded = cache.get(identity)
    torch.testing.assert_close(loaded.single, state.single.half())
    torch.testing.assert_close(loaded.pair, state.pair.half())
    torch.testing.assert_close(loaded.token_mask, state.token_mask)


def test_cache_rejects_schema_mismatch(tmp_path):
    cache = OpenDDEReasoningCache(tmp_path)
    identity = _identity()
    cache.put(identity, _state(identity))
    metadata = cache.path(identity) / "metadata.json"
    text = metadata.read_text().replace('"feature_schema_version": 1', '"feature_schema_version": 9')
    metadata.write_text(text)
    with pytest.raises(RuntimeError, match="cache identity mismatch"):
        cache.get(identity)


def test_cache_put_requires_explicit_overwrite(tmp_path):
    cache = OpenDDEReasoningCache(tmp_path)
    identity = _identity()
    cache.put(identity, _state(identity))
    with pytest.raises(FileExistsError):
        cache.put(identity, _state(identity))
    cache.put(identity, _state(identity), overwrite=True)
