"""Strict provenance loader for the torsion mini_dev architecture."""
from __future__ import annotations

import torch


ARCHITECTURE_ID = "lassodiff_mini_torsion_v2"
SCHEMA_VERSION = 2


def load_mini_v2_checkpoint(path, system=None, *, encoder_name=None, encoder_revision=None,
                            grouped_target_mapping_sha256=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture_id") != ARCHITECTURE_ID or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("checkpoint is not a lassodiff_mini_torsion_v2 checkpoint")
    required = ("source_commit", "stage", "encoder_name", "encoder_revision", "encoder_frozen", "grouped_target_mapping_sha256", "cv_split_manifest_sha256", "source_split_manifest_sha256")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise RuntimeError(f"V2 checkpoint is missing provenance fields: {missing}")
    if checkpoint["encoder_frozen"] is not True:
        raise RuntimeError("V2 checkpoint must record a frozen ESM encoder")
    if encoder_name is not None and checkpoint["encoder_name"] != encoder_name:
        raise RuntimeError("ESM model name mismatch")
    if encoder_revision is not None and checkpoint["encoder_revision"] != encoder_revision:
        raise RuntimeError("ESM revision mismatch")
    if grouped_target_mapping_sha256 is not None and checkpoint["grouped_target_mapping_sha256"] != grouped_target_mapping_sha256:
        raise RuntimeError("grouped target mapping hash mismatch")
    if system is not None:
        if "system" not in checkpoint:
            raise RuntimeError("V2 checkpoint has no unified system state")
        system.load_state_dict(checkpoint["system"], strict=True)
    return checkpoint
