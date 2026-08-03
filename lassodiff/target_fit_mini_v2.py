"""Offline fitting of PDB targets onto the exact Mini V2 decoder manifold."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from .atom_schema_lasso import ATOM_CB, ATOM_CISO, ATOM_N, ATOM_OISO, CandidateCondition
from .chi_geometry import _build_acceptor_reactive_group, build_atom14_from_rigid_groups
from .lasso_core_decoder import decode_lasso_core
from .metrics_mini_v2 import lddt_score
from .torsion_flow import shortest_angular_difference, wrap_angle
from .torsion_state import TorsionState
from .validation.strict_lasso import strict_lasso_check


DECODER_FIT_VERSION = "mini_decoder_fit_v1"


def load_decoder_fit_manifest(cache_root: str | Path) -> dict:
    """Load and cryptographically validate an offline target-fit manifest."""
    path = Path(cache_root) / "manifest.json"
    if not path.is_file():
        raise RuntimeError("decoder-fit cache has no manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.get("manifest_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    expected = hashlib.sha256(canonical).hexdigest()
    if recorded != expected:
        raise RuntimeError("decoder-fit cache manifest SHA256 mismatch")
    if payload.get("fit_version") != DECODER_FIT_VERSION:
        raise RuntimeError("decoder-fit cache version mismatch")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise RuntimeError("decoder-fit cache manifest contains no targets")
    return payload


@dataclass(frozen=True)
class DecoderFitResult:
    backbone_torsions: torch.Tensor
    backbone_mask: torch.Tensor
    chi_angles: torch.Tensor
    chi_mask: torch.Tensor
    core_coordinates: torch.Tensor
    atom14_coordinates: torch.Tensor
    atom14_mask: torch.Tensor
    canonical_ca_rmsd: float
    lddt: float
    strict_valid: bool
    strict_rejection_reasons: tuple[str, ...]
    steps: int
    converged: bool
    fit_version: str = DECODER_FIT_VERSION


def decoder_fit_cache_key(pdb_path: str | Path, candidate: CandidateCondition) -> str:
    path = Path(pdb_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = f"{DECODER_FIT_VERSION}\n{digest}\n{candidate.sequence}\n{candidate.k}\n{candidate.p}".encode()
    return hashlib.sha256(payload).hexdigest()


def save_decoder_fit(path: str | Path, result: DecoderFitResult, *, source: str,
                     candidate: CandidateCondition, cache_key: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "fit_version": result.fit_version,
        "cache_key": cache_key,
        "source": str(source),
        "sequence": candidate.sequence,
        "k": candidate.k,
        "p": candidate.p,
        "result": result,
    }, output)


def load_decoder_fit(path: str | Path, *, candidate: CandidateCondition,
                     expected_key: str) -> DecoderFitResult:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("fit_version") != DECODER_FIT_VERSION or payload.get("cache_key") != expected_key:
        raise RuntimeError("decoder-fit cache provenance mismatch")
    if (payload.get("sequence"), payload.get("k"), payload.get("p")) != (
        candidate.sequence, candidate.k, candidate.p,
    ):
        raise RuntimeError("decoder-fit cache candidate mismatch")
    result = payload.get("result")
    if not isinstance(result, DecoderFitResult):
        raise RuntimeError("decoder-fit cache has no DecoderFitResult")
    return result


def _cosine_angle(left: torch.Tensor, center: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    a = left - center
    b = right - center
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-8)


def _geometry_objective(core: torch.Tensor, sequence: str, candidate: CandidateCondition,
                        acceptor_chi: torch.Tensor, acceptor_chi_mask: torch.Tensor) -> torch.Tensor:
    k = candidate.k
    n0, ciso, oiso = core[0, ATOM_N], core[k, ATOM_CISO], core[k, ATOM_OISO]
    cg, _ciso, _oiso = _build_acceptor_reactive_group(
        sequence=sequence, core=core, acceptor_index=k,
        chi=acceptor_chi, chi_mask=acceptor_chi_mask,
    )
    predecessor = core[k, ATOM_CB] if sequence[k] == "D" else cg
    normal = torch.linalg.cross(oiso - ciso, predecessor - ciso, dim=-1)
    plane = torch.dot(n0 - ciso, normal).abs() / normal.norm().clamp_min(1e-8)
    # The target reactive coordinates already encode valid angles and plane.
    # Keep this term focused on closure; forcing every amide to exactly 120°
    # moves otherwise valid PDB targets away from their measured geometry.
    return 3.0 * ((n0 - ciso).norm() - 1.33).square() + .25 * plane.square()


def fit_decoder_consistent_target(
    *,
    sequence: str,
    candidate: CandidateCondition,
    canonical_core: torch.Tensor,
    core_mask: torch.Tensor,
    atom14_target: torch.Tensor,
    atom14_mask: torch.Tensor,
    aa_ids: torch.Tensor,
    backbone_torsions: torch.Tensor,
    backbone_mask: torch.Tensor,
    chi_angles: torch.Tensor,
    chi_mask: torch.Tensor,
    max_steps: int = 1200,
    learning_rate: float = .03,
) -> DecoderFitResult:
    """Fit target-only labels; never call this function during inference."""
    length = len(sequence)
    if canonical_core.shape != (length, 7, 3):
        raise ValueError("canonical core shape mismatch")
    fitted_backbone = torch.nn.Parameter(backbone_torsions.detach().clone())
    fitted_chi = torch.nn.Parameter(chi_angles.detach().clone())
    initial_backbone = backbone_torsions.detach().clone()
    initial_chi = chi_angles.detach().clone()
    optimizer = torch.optim.Adam((fitted_backbone, fitted_chi), lr=learning_rate)
    token_mask = torch.ones((1, length), dtype=torch.bool, device=canonical_core.device)
    best = None
    best_value = float("inf")
    used_steps = 0
    for step in range(max_steps):
        state = TorsionState(
            fitted_backbone[None, None], backbone_mask[None, None],
            fitted_chi[None, None], chi_mask[None, None],
        )
        decoded = decode_lasso_core(
            state, sequences=[sequence], candidates=[candidate], token_mask=token_mask,
        )[0, 0]
        valid = core_mask[..., None].to(decoded.dtype)
        coordinate = ((decoded - canonical_core).square() * valid).sum() / valid.sum().clamp_min(1)
        ca = (decoded[:, 1] - canonical_core[:, 1]).square().sum(-1).mean()
        reactive = (decoded[candidate.k, ATOM_CISO:ATOM_OISO + 1]
                    - canonical_core[candidate.k, ATOM_CISO:ATOM_OISO + 1]).square().mean()
        geometry = _geometry_objective(
            decoded, sequence, candidate, fitted_chi[candidate.k], chi_mask[candidate.k],
        )
        regularizer = shortest_angular_difference(fitted_backbone, initial_backbone).square()
        regularizer = (regularizer * backbone_mask).sum() / backbone_mask.sum().clamp_min(1)
        chi_regularizer = shortest_angular_difference(fitted_chi, initial_chi).square()
        chi_regularizer = (chi_regularizer * chi_mask).sum() / chi_mask.sum().clamp_min(1)
        loss = ca + 5.0 * reactive + geometry + .0001 * (regularizer + chi_regularizer)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            fitted_backbone.copy_(wrap_angle(fitted_backbone) * backbone_mask)
            fitted_chi.copy_(wrap_angle(fitted_chi) * chi_mask)
            value = float(loss.detach())
            if value < best_value:
                best_value = value
                best = (fitted_backbone.detach().clone(), fitted_chi.detach().clone())
            used_steps = step + 1
            if step >= 200 and value < 2e-3:
                break
    if best is None:
        raise RuntimeError("decoder target fitting produced no finite state")
    fitted_backbone_value, fitted_chi_value = best
    state = TorsionState(
        fitted_backbone_value[None, None], backbone_mask[None, None],
        fitted_chi_value[None, None], chi_mask[None, None],
    )
    decoded = decode_lasso_core(
        state, sequences=[sequence], candidates=[candidate], token_mask=token_mask,
    )[0, 0]
    decoded_atom14, decoded_atom14_mask = build_atom14_from_rigid_groups(
        decoded, aa_ids, fitted_chi_value, chi_mask, candidate,
    )
    strict = strict_lasso_check(decoded, candidate.core_atom_mask.to(decoded.device), candidate)
    ca_rmsd = torch.sqrt((decoded[:, 1] - canonical_core[:, 1]).square().sum(-1).mean())
    score = lddt_score(decoded[:, 1], canonical_core[:, 1])
    converged = bool(strict.valid and ca_rmsd < .75 and score > .95)
    return DecoderFitResult(
        fitted_backbone_value, backbone_mask.detach().clone(), fitted_chi_value,
        chi_mask.detach().clone(), decoded.detach(), decoded_atom14.detach(),
        decoded_atom14_mask.detach(), float(ca_rmsd), float(score), bool(strict.valid),
        tuple(strict.rejection_reasons), used_steps, converged,
    )
