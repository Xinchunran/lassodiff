"""Strict provenance loader for the torsion mini_dev architecture."""
from __future__ import annotations

import torch


ARCHITECTURE_ID = "lassodiff_mini_torsion_v2"
SCHEMA_VERSION = 2


def save_mini_v2_checkpoint(path, *, system, optimizer=None, step=0, stage="backbone",
                            world_size=1, preflight=None, provenance=None):
    if getattr(system, "architecture_id", ARCHITECTURE_ID) != ARCHITECTURE_ID:
        raise RuntimeError("cannot save a non-V2 system as a V2 checkpoint")
    payload = {
        "architecture_id": ARCHITECTURE_ID, "schema_version": SCHEMA_VERSION,
        "stage": stage, "system": system.state_dict(), "step": int(step),
        "world_size": int(world_size), "preflight": preflight or {},
        **(provenance or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return payload


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


def validate_resume_checkpoint(checkpoint, *, stage, cv_split_manifest_sha256=None,
                               source_split_manifest_sha256=None, dataset_mapping_sha256=None,
                               encoder_name=None, encoder_revision=None,
                               esm_cache_manifest_sha256=None):
    if checkpoint.get("architecture_id") != ARCHITECTURE_ID or checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("resume checkpoint is not V2")
    if checkpoint.get("stage") != stage:
        raise RuntimeError("resume stage mismatch")
    provenance = checkpoint
    for key, expected in (("cv_split_manifest_sha256", cv_split_manifest_sha256),
                          ("source_split_manifest_sha256", source_split_manifest_sha256),
                          ("dataset_mapping_sha256", dataset_mapping_sha256),
                          ("encoder_name", encoder_name), ("encoder_revision", encoder_revision),
                          ("esm_cache_manifest_sha256", esm_cache_manifest_sha256)):
        if expected is not None and provenance.get(key) != expected:
            raise RuntimeError(f"resume provenance mismatch: {key}")
    if "optimizer" not in checkpoint:
        raise RuntimeError("resume checkpoint has no optimizer state")
