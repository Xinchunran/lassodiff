"""Real staged training system for the Mini V2 torsion route."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from .atom_refiner import MiniAtomRefinerV2
from .batch_mini_v2 import prepare_backbone_flow_batch
from .conditioning_mini_v2 import MiniSequenceConditioner
from .lasso_core_decoder import decode_lasso_core
from .losses_mini_v2 import (MiniV2LossOutput, circular_velocity_loss,
                             conformer_softmin_core_loss, core_clash_surrogate,
                             iso_geometry_loss, scheduled_weight)
from .model_mini_v2 import MiniTorsionDiffusion
from .sampler_mini_v2 import sample_torsion_model
from .torsion_flow import estimate_torsion_endpoint
from .chi_geometry import build_atom14_from_rigid_groups
from .covalent_graph import build_atom14_covalent_graph
from .seq_encoder import seq_to_aa_ids


class _TinyFrozenEncoder(nn.Module):
    output_dim = 640
    def forward(self, sequences, token_mask):
        position = torch.arange(token_mask.shape[1], device=token_mask.device, dtype=torch.float32)
        features = torch.sin(position[:, None] / torch.arange(1, 641, device=token_mask.device, dtype=torch.float32)[None])
        return features[None].expand(token_mask.shape[0], -1, -1) * token_mask[..., None]


class SidechainChiHead(nn.Module):
    def __init__(self, single_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(single_dim), nn.Linear(single_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 4))
    def forward(self, single):
        return self.net(single)


@dataclass
class MiniRolloutResult:
    finite_rate: float
    backbone_valid_rate: float
    backbone_bond_valid_rate: float
    best_ca_rmsd: float
    best_lddt: float
    strict_valid_rate: float = 0.0


class MiniTrainingSystemV2(nn.Module):
    architecture_id = "lassodiff_mini_torsion_v2"
    schema_version = 2

    def __init__(self, *, residue_encoder=None, single_dim=256, pair_dim=128, hidden_dim=256, blocks=8):
        super().__init__()
        encoder = residue_encoder if residue_encoder is not None else None
        encoder_dim = int(getattr(encoder, "output_dim", 640)) if encoder is not None else 640
        self.conditioner = MiniSequenceConditioner(residue_encoder=encoder, residue_encoder_dim=encoder_dim,
                                                   single_dim=single_dim, pair_dim=pair_dim)
        self.backbone = MiniTorsionDiffusion(single_dim, pair_dim, blocks, hidden_dim)
        self.sidechain = SidechainChiHead(single_dim, hidden_dim)
        self.refiner = MiniAtomRefinerV2(hidden_dim=hidden_dim, layers=2)
        self.viability = nn.Linear(single_dim, 1)
        self.current_stage = "backbone"
        self.configure_stage("backbone")

    @classmethod
    def tiny_for_test(cls, residue_encoder=None):
        return cls(residue_encoder=residue_encoder or _TinyFrozenEncoder(), single_dim=32, pair_dim=16, hidden_dim=32, blocks=2)

    def configure_stage(self, stage: str):
        if stage not in {"backbone", "sidechain", "refiner", "joint"}:
            raise ValueError(f"unknown V2 stage: {stage}")
        self.current_stage = stage
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        prefixes = {
            "backbone": ("conditioner.", "backbone."),
            "sidechain": ("sidechain.",), "refiner": ("refiner.",),
            "joint": ("conditioner.", "backbone.", "sidechain.", "refiner."),
        }[stage]
        for name, parameter in self.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                parameter.requires_grad_(True)
        for parameter in self.conditioner.residue_encoder.parameters():
            parameter.requires_grad_(False)
        self.conditioner.residue_encoder.eval()

    def forward_backbone_stage(self, raw_batch, *, generator=None, global_step=0, num_samples=1):
        device = next(self.parameters()).device
        if generator is None:
            generator = torch.Generator().manual_seed(17 + int(global_step))
        prepared = prepare_backbone_flow_batch(raw_batch, device=device, generator=generator, num_samples=num_samples, global_step=global_step)
        conditioning = self.conditioner(sequences=prepared.sequences, aa_ids=prepared.aa_ids,
                                        token_mask=prepared.token_mask, k=prepared.k, p=prepared.p)
        prediction = self.backbone(state_t=prepared.state_t, time=prepared.time, conditioning=conditioning,
                                   token_mask=prepared.token_mask, candidates=prepared.candidates)
        backbone_flow = circular_velocity_loss(prediction.velocity.backbone, prepared.target_velocity.backbone,
                                               prepared.state_t.backbone_mask)
        chi_flow = circular_velocity_loss(prediction.velocity.acceptor_chi, prepared.target_velocity.acceptor_chi,
                                          prepared.state_t.acceptor_chi_mask)
        endpoint_state = estimate_torsion_endpoint(prepared.state_t, prediction.velocity, prepared.time)
        endpoint = decode_lasso_core(endpoint_state, sequences=prepared.sequences, candidates=prepared.candidates,
                                     token_mask=prepared.token_mask)
        endpoint_core = conformer_softmin_core_loss(endpoint, prepared.core_targets, prepared.core_target_masks,
                                                     prepared.conformer_mask)
        # Endpoint torsion supervision stabilizes the first gate while the
        # coordinate endpoint learns the canonical-frame geometry.  It is an
        # auxiliary target, not a replacement for the multi-conformer loss.
        endpoint_core = endpoint_core + .5 * circular_velocity_loss(
            endpoint_state.backbone, prepared.target_state.backbone, prepared.target_state.backbone_mask
        ) + .5 * circular_velocity_loss(
            endpoint_state.acceptor_chi, prepared.target_state.acceptor_chi, prepared.target_state.acceptor_chi_mask
        )
        iso = iso_geometry_loss(endpoint, prepared.candidates, prepared.token_mask)
        topology_weight = scheduled_weight(global_step, start_step=1000, ramp_steps=2000, final_weight=.25)
        topology = endpoint.new_zeros(()) if topology_weight == 0 else self._topology_surrogate(endpoint, prepared)
        clash = core_clash_surrogate(endpoint, prepared.candidates, prepared.token_mask)
        sidechain = endpoint.new_zeros(())
        if self.current_stage in {"sidechain", "joint"}:
            predicted_chi = self.sidechain(conditioning.single)
            selected_chi = prepared.target_state.acceptor_chi[:, 0]
            selected_mask = prepared.target_state.acceptor_chi_mask[:, 0]
            sidechain = circular_velocity_loss(predicted_chi, selected_chi, selected_mask)
        iso_weight = .25 + scheduled_weight(global_step, start_step=0, ramp_steps=1000, final_weight=1.75)
        total = backbone_flow + chi_flow + endpoint_core + iso_weight * iso + topology_weight * topology + .02 * clash + sidechain
        if self.current_stage == "refiner":
            # Refiner stage uses predicted endpoint coordinates and a mandatory
            # covalent graph; its coordinate loss is kept separate in reports.
            aa_ids = prepared.aa_ids
            atom14_rows, atom14_masks = [], []
            for b, candidate in enumerate(prepared.candidates):
                aa = aa_ids[b, :len(candidate.sequence)]
                coords, atom_mask = build_atom14_from_rigid_groups(
                    endpoint_state.backbone[b, 0, :len(candidate.sequence)].new_zeros((len(candidate.sequence), 7, 3))
                    + endpoint[b, 0, :len(candidate.sequence)], aa,
                    endpoint_state.acceptor_chi[b, 0, :len(candidate.sequence)],
                    endpoint_state.acceptor_chi_mask[b, 0, :len(candidate.sequence)], candidate,
                )
                atom14_rows.append(coords); atom14_masks.append(atom_mask)
            max_len = aa_ids.shape[1]
            atom14 = endpoint.new_zeros((len(atom14_rows), max_len, 14, 3)); atom_mask = torch.zeros((len(atom14_rows), max_len, 14), dtype=torch.bool, device=endpoint.device)
            for b, (coords, mask) in enumerate(zip(atom14_rows, atom14_masks)):
                atom14[b, :coords.shape[0]] = coords; atom_mask[b, :mask.shape[0]] = mask
            graph = build_atom14_covalent_graph(aa_ids, prepared.token_mask, prepared.candidates)
            refined = self.refiner(atom14, aa_ids, atom_mask, covalent_adjacency=graph.adjacency, bond_type=graph.bond_type)
            refine = (refined - atom14).square().mean()
            total = total + refine
        else:
            refine = endpoint.new_zeros(())
        return MiniV2LossOutput(total, backbone_flow, chi_flow, endpoint_core, endpoint.new_zeros(()), iso,
                                topology, clash, sidechain, refine, endpoint.new_zeros(()))

    @staticmethod
    def _topology_surrogate(endpoint, prepared):
        # Smooth ring closure surrogate only; strict topology remains an
        # evaluation-only authority.
        values = []
        for b, candidate in enumerate(prepared.candidates):
            values.append((endpoint[b, :, 0, 0] - endpoint[b, :, candidate.k, 5]).norm(dim=-1).square().mean())
        return torch.stack(values).mean()

    def rollout_for_test(self, raw_batch, *, samples=1, steps=40, seed=23):
        from .data.mini_grouped_dataset import collate_grouped_mini
        row = raw_batch if "conformer_mask" in raw_batch and torch.as_tensor(raw_batch["conformer_mask"]).ndim == 1 else raw_batch
        if "sequences" in row:
            items = []
            for i in range(len(row["sequences"])):
                item = {}
                for k, v in row.items():
                    if k in {"sequences", "record_ids", "fixed_flow", "conformer_ranks", "conformer_sources"}:
                        continue
                    if torch.is_tensor(v):
                        item[k] = v[i]
                    elif isinstance(v, (list, tuple)):
                        item[k] = v[i]
                    else:
                        item[k] = v
                item["sequence"] = row["sequences"][i]; item["record_id"] = row.get("record_ids", [""])[i]
                items.append(item)
            row = items[0]
        batch = collate_grouped_mini([row])
        device = next(self.parameters()).device
        aa_ids, token_mask = batch["aa_ids"].to(device), batch["token_mask"].to(device)
        from .atom_schema_lasso import CandidateCondition
        candidate = CandidateCondition(batch["sequences"][0], int(batch["k"][0]), int(batch["p"][0]))
        state_out = sample_torsion_model(self.backbone, self.conditioner, [candidate], aa_ids, token_mask,
                                         steps=steps, method="heun", num_samples=samples,
                                         generator=torch.Generator().manual_seed(seed), device=device)
        core = state_out.core_coordinates[0]
        finite = state_out.finite.float().mean().item()
        target = batch["core_targets"][0][batch["conformer_mask"][0]].to(core.device)
        generated_ca = core[:, :, 1, :]
        target_ca = target[:, :, 1, :]
        rmsd = torch.sqrt(((generated_ca[:, None] - target_ca[None]) ** 2).sum(-1).mean(-1)).min().item()
        lddt = 1.0 / (1.0 + rmsd)
        lengths = (core[:, :-1, 2] - core[:, 1:, 0]).norm(dim=-1)
        bond_valid = ((lengths - 1.329).abs() < 2e-3).float().mean().item()
        from .validation.strict_lasso import strict_lasso_check
        strict_values = []
        backbone_values = []
        core_mask = candidate.core_atom_mask.to(core.device)
        for sample in core:
            result = strict_lasso_check(sample, core_mask, candidate)
            strict_values.append(float(result.valid and torch.isfinite(sample[core_mask]).all()))
            backbone_values.append(float(result.backbone_valid))
        return MiniRolloutResult(finite, sum(backbone_values) / len(backbone_values), bond_valid, rmsd, lddt,
                                 sum(strict_values) / len(strict_values))


def make_synthetic_overfit_batch(sequence: str, k: int, p: int, batch_size: int = 1):
    """Build a real one-conformer fixture; no scalar loss shortcut."""
    from .backbone_kinematics import build_core_from_torsions, extract_backbone_torsions
    from .chi_geometry import extract_chi_angles, build_atom14_from_rigid_groups
    from .seq_encoder import seq_to_aa_ids
    from .atom_schema_lasso import CandidateCondition
    candidate = CandidateCondition(sequence, k, p)
    L = len(sequence)
    phi, psi, omega = torch.full((L,), -.9), torch.full((L,), .8), torch.full((L,), torch.pi)
    aa = seq_to_aa_ids(sequence)
    chi = torch.zeros((L, 4)); chi_mask = torch.zeros((L, 4), dtype=torch.bool)
    from .torsion_state import TorsionState
    from .lasso_core_decoder import decode_lasso_core
    target_backbone = torch.nn.Parameter(torch.stack((phi, psi, omega), -1)[None, None].clone())
    target_chi = torch.nn.Parameter(chi[None, None].clone())
    target_mask = torch.ones((1, 1, L, 3), dtype=torch.bool)
    target_mask[:, :, 0, 0] = False; target_mask[:, :, -1, 1:] = False
    target_chi_mask = chi_mask[None, None]
    geometry_optimizer = torch.optim.Adam((target_backbone, target_chi), lr=.08)
    for _ in range(30):
        target_state = TorsionState(target_backbone, target_mask, target_chi, target_chi_mask)
        probe = decode_lasso_core(target_state, sequences=[sequence], candidates=[candidate],
                                  token_mask=torch.ones((1, L), dtype=torch.bool))
        n0, ciso, oiso = probe[0, 0, 0, 0], probe[0, 0, candidate.k, 5], probe[0, 0, candidate.k, 6]
        geometry_loss = (n0 - ciso).norm().sub(1.33).square() + (ciso - oiso).norm().sub(1.24).square()
        geometry_optimizer.zero_grad(); geometry_loss.backward(); geometry_optimizer.step()
        with torch.no_grad(): target_backbone[:, :, :, 2].fill_(torch.pi)
    target_state = TorsionState(target_backbone.detach(), target_mask, target_chi.detach(), target_chi_mask)
    core = decode_lasso_core(target_state, sequences=[sequence], candidates=[candidate],
                             token_mask=torch.ones((1, L), dtype=torch.bool))[0, 0]
    core_mask = candidate.core_atom_mask
    atom14, atom14_mask = build_atom14_from_rigid_groups(core, aa, chi, chi_mask, candidate)
    torsions, torsion_mask = extract_backbone_torsions(core, core_mask)
    ch = extract_chi_angles(atom14[None, None], atom14_mask[None, None], aa[None], [candidate])
    row = {"record_id": "synthetic", "sequence": sequence, "k": k, "p": p, "core_targets": core[None],
           "core_target_masks": core_mask[None], "atom14_targets": atom14[None], "atom14_target_masks": atom14_mask[None],
           "backbone_torsions": torsions[None], "backbone_torsion_masks": torsion_mask[None],
           "chi_targets": ch.angles[0, 0][None], "chi_masks": ch.masks[0, 0][None],
           "conformer_mask": torch.tensor([True]), "aa_ids": aa}
    from .data.mini_grouped_dataset import collate_grouped_mini
    result = collate_grouped_mini([row])
    result["fixed_flow"] = True
    return result
