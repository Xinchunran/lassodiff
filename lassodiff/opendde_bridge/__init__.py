"""Fail-closed bridge from pinned OpenDDE residue reasoning to LassoDiff V3."""

from .schema import FEATURE_SCHEMA_VERSION, OpenDDEReasoningState, validate_reasoning_state

__all__ = ["FEATURE_SCHEMA_VERSION", "OpenDDEReasoningState", "validate_reasoning_state"]
