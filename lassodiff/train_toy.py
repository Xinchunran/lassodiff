from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
import sys
import time
from datetime import datetime
import glob
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lassodiff.esm_embedder import ESMEmbedder, ESMConfig
from lassodiff.lasso_features import build_lasso_features
from lassodiff.seq_encoder import SequenceEncoder, SeqEncConfig
from lassodiff.config import sinusoidal_pos1d
from lassodiff.pair_init import PairInitializer, PairInitConfig
from lassodiff.conditioner_pairformer import PairformerConditioner, PairformerCfg
from lassodiff.conditioner_simplefold import SimpleFoldConditioner, SimpleFoldCfg
from lassodiff.score_head_protenix_residue import ProtenixResidueScoreHead, ScoreCfg
from lassodiff.flow_matching import SigmaSchedule, sample_t, flow_interpolate
from lassodiff.losses_lasso import (
    LossWeights,
    IsoCfg,
    PlugCfg,
    LinkCfg,
    TubeCfg,
    ATOM_CA,
    region_weights,
    loss_flow,
    loss_bb,
    loss_iso_bb,
    project_iso_bb,
    loss_plug_thread_ca,
    gauss_linking_integral_ca,
    loss_link_ca,
    loss_ring_tube_ca,
    loss_clash_bb,
)
from lassodiff.multimodal import softmin_aggregate
from lassodiff.dataset_toy import ToyLassoDataset, collate as collate_toy
from lassodiff.dataset_lasso import LassoPDBDataset, split_by_sequence_identity, collate as collate_lasso


class LassoDiffModel(nn.Module):
    def __init__(self, mode: str = "pairformer"):
        super().__init__()
        self.mode = mode
        self.w_link_schedule = None
        self.esm = ESMEmbedder(ESMConfig(model_name="esm2_t30_150M_UR50D", freeze=True))
        self.seq_enc = SequenceEncoder(
            esm_dim=self.esm.esm_dim,
            lasso_feat_dim=7,
            cfg=SeqEncConfig(c_s_inputs=256),
        )
        self.pair_init = PairInitializer(c_s_inputs=256, cfg=PairInitConfig(c_z=128))
        if mode == "pairformer":
            self.cond = PairformerConditioner(
                c_s_inputs=256,
                cfg=PairformerCfg(n_blocks=4, n_heads=8, c_s=384, c_z=128),
            )
        else:
            self.cond = SimpleFoldConditioner(
                c_s_inputs=256,
                cfg=SimpleFoldCfg(c_s=384, n_layers=4, n_heads=8),
            )
        self.score = ProtenixResidueScoreHead(
            ScoreCfg(c_s_inputs=256, c_s=384, c_z=128, c_a=128, n_blocks=4, n_heads=8)
        )

    def forward(self, seqs, aa_ids, token_mask, k, p, x_t, sigma, t_scalar):
        B, Ns, L, _A, _ = x_t.shape
        device = x_t.device
        esm_out = self.esm(seqs, return_pooled=False)
        esm_res = esm_out["esm_residue"]
        if esm_res.shape[1] != L:
            raise ValueError(f"ESM length {esm_res.shape[1]} != L {L}")
        lasso_feats = torch.stack(
            [build_lasso_features(L, int(k[i].item()), int(p[i].item()), device) for i in range(B)],
            dim=0,
        )
        pos = sinusoidal_pos1d(L, 32, device).unsqueeze(0).repeat(B, 1, 1)
        s_inputs = self.seq_enc(aa_ids, esm_res, lasso_feats, pos)
        z_init, pair_mask = self.pair_init(s_inputs, token_mask, k, p)
        if self.mode == "pairformer":
            s_trunk, z_trunk = self.cond(s_inputs, z_init, pair_mask)
        else:
            s_trunk, z_trunk = self.cond(s_inputs, z_init, token_mask, t_scalar)
        v_pred = self.score(
            x_t=x_t,
            sigma=sigma,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            token_mask=token_mask,
        )
        return v_pred


def rmsd_ca(x1_pred, x1_true, token_mask):
    diff2 = (x1_pred - x1_true) ** 2
    mask = token_mask[:, None, :, None].float()
    denom = mask.sum() * 3.0 + 1e-8
    return torch.sqrt((diff2 * mask).sum() / denom)


def lddt_ca(x1_pred, x1_true, token_mask, cutoff=15.0):
    B, Ns, L, _ = x1_pred.shape
    mask = token_mask.float()
    xi = x1_pred[:, :, :, None, :]
    xj = x1_pred[:, :, None, :, :]
    yi = x1_true[:, :, :, None, :]
    yj = x1_true[:, :, None, :, :]
    d_pred = torch.linalg.vector_norm(xi - xj, dim=-1)
    d_true = torch.linalg.vector_norm(yi - yj, dim=-1)
    eye = torch.eye(L, device=x1_pred.device).bool()
    valid = (mask[:, None, :, None] * mask[:, None, None, :]).bool()
    valid = valid & (~eye[None, None, :, :])
    valid = valid & (d_true < cutoff)
    diff = torch.abs(d_pred - d_true)
    score = (
        (diff < 0.5).float()
        + (diff < 1.0).float()
        + (diff < 2.0).float()
        + (diff < 4.0).float()
    ) / 4.0
    num = (score * valid.float()).sum()
    denom = valid.float().sum().clamp(min=1.0)
    return num / denom


def contact_precision(x1_pred, x1_true, token_mask, cutoff=8.0, min_sep=3):
    B, Ns, L, _ = x1_pred.shape
    mask = token_mask.float()
    xi = x1_pred[:, :, :, None, :]
    xj = x1_pred[:, :, None, :, :]
    yi = x1_true[:, :, :, None, :]
    yj = x1_true[:, :, None, :, :]
    d_pred = torch.linalg.vector_norm(xi - xj, dim=-1)
    d_true = torch.linalg.vector_norm(yi - yj, dim=-1)
    idx = torch.arange(L, device=x1_pred.device)
    sep = (idx[None, :, None] - idx[None, None, :]).abs()
    sep_mask = sep >= min_sep
    valid = (mask[:, None, :, None] * mask[:, None, None, :]).bool()
    valid = valid & sep_mask[None, None, :, :]
    eye = torch.eye(L, device=x1_pred.device).bool()
    valid = valid & (~eye[None, None, :, :])
    d_pred = d_pred.masked_fill(~valid, 1e8)
    d_true = d_true.masked_fill(~valid, 1e8)
    k = max(L, 1)
    d_pred_flat = d_pred.reshape(B * Ns, L * L)
    d_true_flat = d_true.reshape(B * Ns, L * L)
    topk = torch.topk(-d_pred_flat, k=k, dim=-1).indices
    pred_contacts = torch.gather(d_pred_flat, 1, topk) < cutoff
    true_contacts = torch.gather(d_true_flat, 1, topk) < cutoff
    tp = (pred_contacts & true_contacts).float().sum()
    denom = pred_contacts.float().sum().clamp(min=1.0)
    return tp / denom


def run_epoch(
    model,
    dl,
    device,
    mode,
    opt,
    sched,
    w,
    iso_cfg,
    plug_cfg,
    link_cfg,
    tube_cfg,
    hard_iso,
):
    is_train = mode == "train"
    model.train(is_train)
    totals = {
        "loss": 0.0,
        "flow": 0.0,
        "bb": 0.0,
        "iso": 0.0,
        "plug": 0.0,
        "thr": 0.0,
        "clash": 0.0,
        "link_loss": 0.0,
        "gli_pred": 0.0,
        "gli_gt": 0.0,
        "tube": 0.0,
        "rmsd": 0.0,
        "lddt": 0.0,
        "cprec": 0.0,
        "steps": 0,
        "tokens": 0,
        "time": 0.0,
    }
    for step, batch in enumerate(dl):
        t0 = time.time()
        aa_ids = batch["aa_ids"].to(device)
        token_mask = batch["token_mask"].to(device)
        k = batch["k"].to(device)
        p = batch["p"].to(device)
        coords_list = [c.to(device) for c in batch["coords_list"]]
        atom_mask_list = [m.to(device) for m in batch["atom_mask_list"]]
        iso_acceptor_index = batch["iso_acceptor_index"].to(device)
        for i in range(len(coords_list)):
            if not torch.isfinite(coords_list[i]).all():
                coords_list[i] = torch.nan_to_num(coords_list[i], nan=0.0, posinf=1e4, neginf=-1e4)
        B, L, A, _ = coords_list[0].shape
        Ns = 1

        x0 = torch.randn(B, Ns, L, A, 3, device=device) * 5.0
        t = sample_t(B, Ns, device=device)
        sigma = sched.sigma(t)

        w_res = torch.stack([region_weights(L, int(k[i]), int(p[i]), device) for i in range(B)], dim=0)

        losses = []
        flow_terms = []
        bb_terms = []
        iso_terms = []
        plug_terms = []
        thr_terms = []
        clash_terms = []
        link_loss_terms = []
        gli_pred_terms = []
        gli_gt_terms = []
        tube_terms = []
        rmsd_terms = []
        lddt_terms = []
        cprec_terms = []
        seqs = batch["seqs"]
        for x1_true, atom_mask in zip(coords_list, atom_mask_list):
            x1_true = x1_true.unsqueeze(1)
            x_t, v_star = flow_interpolate(x0, x1_true, t)

            v_pred = model(
                seqs=seqs,
                aa_ids=aa_ids,
                token_mask=token_mask,
                k=k,
                p=p,
                x_t=x_t,
                sigma=sigma,
                t_scalar=t,
            )
            if not torch.isfinite(v_pred).all():
                continue

            x1_pred = x_t + (1.0 - t[:, :, None, None, None]) * v_pred
            if hard_iso:
                x1_pred = project_iso_bb(x1_pred, iso_acceptor_index, token_mask, atom_mask, iso_cfg=iso_cfg)

            L_flow = loss_flow(v_pred, v_star, token_mask, w_res=w_res, atom_mask=atom_mask)
            L_bb = loss_bb(x1_pred, x1_true, token_mask, w_res=w_res, atom_mask=atom_mask)
            L_iso = loss_iso_bb(x1_pred, iso_acceptor_index, token_mask, atom_mask, iso_cfg=iso_cfg)
            x1_pred_ca = x1_pred[..., ATOM_CA, :]
            x1_true_ca = x1_true[..., ATOM_CA, :]
            L_plug, L_thr = loss_plug_thread_ca(x1_pred_ca, k, p, token_mask, plug_cfg=plug_cfg)
            L_clash = loss_clash_bb(x1_pred, token_mask, atom_mask, iso_acceptor_index)
            gli_pred = gauss_linking_integral_ca(x1_pred_ca, k, p, token_mask, link_cfg=link_cfg).mean()
            gli_gt = gauss_linking_integral_ca(x1_true_ca, k, p, token_mask, link_cfg=link_cfg).mean()
            L_link = loss_link_ca(x1_pred_ca, k, p, token_mask, gli_gt=gli_gt, link_cfg=link_cfg) if w.w_link != 0.0 else x1_pred.new_tensor(0.0)
            L_tube = loss_ring_tube_ca(x1_pred_ca, k, p, token_mask, tube_cfg=tube_cfg) if w.w_tube != 0.0 else x1_pred.new_tensor(0.0)
            L_rmsd = rmsd_ca(x1_pred_ca, x1_true_ca, token_mask)
            L_lddt = lddt_ca(x1_pred_ca, x1_true_ca, token_mask)
            L_cprec = contact_precision(x1_pred_ca, x1_true_ca, token_mask)

            total = (
                w.w_flow * L_flow
                + w.w_bb * L_bb
                + w.w_iso * L_iso
                + w.w_plug * L_plug
                + w.w_thr * L_thr
                + w.w_clash * L_clash
                + w.w_link * L_link
                + w.w_tube * L_tube
            )
            losses.append(total)
            flow_terms.append(L_flow)
            bb_terms.append(L_bb)
            iso_terms.append(L_iso)
            plug_terms.append(L_plug)
            thr_terms.append(L_thr)
            clash_terms.append(L_clash)
            link_loss_terms.append(L_link)
            gli_pred_terms.append(gli_pred)
            gli_gt_terms.append(gli_gt)
            tube_terms.append(L_tube)
            rmsd_terms.append(L_rmsd)
            lddt_terms.append(L_lddt)
            cprec_terms.append(L_cprec)

        if len(losses) == 0:
            continue
        L = softmin_aggregate(torch.stack(losses), tau=0.5)
        flow_v = torch.stack(flow_terms).mean()
        bb_v = torch.stack(bb_terms).mean()
        iso_v = torch.stack(iso_terms).mean()
        plug_v = torch.stack(plug_terms).mean()
        thr_v = torch.stack(thr_terms).mean()
        clash_v = torch.stack(clash_terms).mean()
        link_loss_v = torch.stack(link_loss_terms).mean()
        gli_pred_v = torch.stack(gli_pred_terms).mean()
        gli_gt_v = torch.stack(gli_gt_terms).mean()
        tube_v = torch.stack(tube_terms).mean()
        rmsd_v = torch.stack(rmsd_terms).mean()
        lddt_v = torch.stack(lddt_terms).mean()
        cprec_v = torch.stack(cprec_terms).mean()

        if not torch.isfinite(L):
            continue
        if is_train:
            opt.zero_grad(set_to_none=True)
            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        totals["loss"] += L.item()
        totals["flow"] += flow_v.item()
        totals["bb"] += bb_v.item()
        totals["iso"] += iso_v.item()
        totals["plug"] += plug_v.item()
        totals["thr"] += thr_v.item()
        totals["clash"] += clash_v.item()
        totals["link_loss"] += link_loss_v.item()
        totals["gli_pred"] += gli_pred_v.item()
        totals["gli_gt"] += gli_gt_v.item()
        totals["tube"] += tube_v.item()
        totals["rmsd"] += rmsd_v.item()
        totals["lddt"] += lddt_v.item()
        totals["cprec"] += cprec_v.item()
        totals["steps"] += 1
        totals["tokens"] += int(token_mask.sum().item())
        totals["time"] += time.time() - t0

    steps = max(totals["steps"], 1)
    return {
        "loss": totals["loss"] / steps,
        "flow": totals["flow"] / steps,
        "bb": totals["bb"] / steps,
        "iso": totals["iso"] / steps,
        "plug": totals["plug"] / steps,
        "thr": totals["thr"] / steps,
        "clash": totals["clash"] / steps,
        "link_loss": totals["link_loss"] / steps,
        "gli_pred": totals["gli_pred"] / steps,
        "gli_gt": totals["gli_gt"] / steps,
        "tube": totals["tube"] / steps,
        "rmsd": totals["rmsd"] / steps,
        "lddt": totals["lddt"] / steps,
        "cprec": totals["cprec"] / steps,
        "time_s": totals["time"],
        "steps": totals["steps"],
        "tokens": totals["tokens"],
    }


def _dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    return 0, 1, int(os.environ.get("LOCAL_RANK", 0))


def _setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def _dist_barrier(local_rank: int, device: torch.device):
    if not (dist.is_available() and dist.is_initialized()):
        return
    if device.type == "cuda":
        torch.cuda.synchronize(local_rank)
    dist.barrier()


def _reduce_totals(totals: dict, device: torch.device) -> dict:
    if not (dist.is_available() and dist.is_initialized()):
        return totals
    avg_keys = [
        "loss",
        "flow",
        "bb",
        "iso",
        "plug",
        "thr",
        "clash",
        "link_loss",
        "gli_pred",
        "gli_gt",
        "tube",
        "rmsd",
        "lddt",
        "cprec",
    ]
    steps = float(totals["steps"])
    vec = torch.tensor([float(totals[k]) * steps for k in avg_keys] + [steps, float(totals["tokens"])], device=device)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    t = torch.tensor([float(totals["time_s"])], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    total_steps = float(vec[len(avg_keys)].item())
    denom = total_steps if total_steps > 0 else 1.0
    for i, k in enumerate(avg_keys):
        totals[k] = float(vec[i].item()) / denom
    totals["steps"] = int(total_steps)
    totals["tokens"] = int(vec[len(avg_keys) + 1].item())
    totals["time_s"] = float(t.item())
    return totals


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _log_line(msg: str, fp: Path | None):
    if fp is not None:
        with fp.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
    print(msg)


def _aa1_to_aa3():
    return {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "Y": "TYR",
    }


def _write_backbone_pdb(
    path: Path,
    seq: str,
    coords: torch.Tensor,
    atom_mask: torch.Tensor,
    iso_acceptor_index: int | None = None,
    iso_acceptor_type: str | None = None,
):
    aa1_to_aa3 = _aa1_to_aa3()
    atom_count = 1
    iso_type = iso_acceptor_type.strip().upper() if iso_acceptor_type else ""
    with path.open("w", encoding="utf-8") as f:
        L, A, _ = coords.shape
        for i in range(L):
            resname = aa1_to_aa3.get(seq[i], "UNK")
            resseq = i + 1
            for a in range(A):
                if not bool(atom_mask[i, a].item()):
                    continue
                if a == 0:
                    atom_name = "N"
                elif a == 1:
                    atom_name = "CA"
                elif a == 2:
                    atom_name = "C"
                elif a == 3:
                    atom_name = "O"
                elif i == iso_acceptor_index and iso_type == "ASP":
                    atom_name = ["CG", "OD1", "OD2"][a - 4]
                elif i == iso_acceptor_index and iso_type == "GLU":
                    atom_name = ["CD", "OE1", "OE2"][a - 4]
                else:
                    atom_name = f"C{a}"
                x, y, z = coords[i, a].tolist()
                element = atom_name[0]
                f.write(
                    f"ATOM  {atom_count:5d} {atom_name:>4} {resname:>3} A{resseq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2}\n"
                )
                atom_count += 1
        f.write("END\n")


def _export_val_pdbs(
    model,
    ds,
    val_idx,
    device,
    sched,
    hard_iso,
    export_dir: Path,
    n_export: int,
    structure_dir: str | None,
    seed: int,
):
    if not isinstance(ds, LassoPDBDataset):
        return
    _ensure_dir(export_dir)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    collate_fn = _select_collate(ds)
    dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    torch.manual_seed(seed)
    exported = 0
    model.eval()
    for batch in dl:
        if exported >= n_export:
            break
        names = batch["names"]
        seqs = batch["seqs"]
        aa_ids = batch["aa_ids"].to(device)
        token_mask = batch["token_mask"].to(device)
        k = batch["k"].to(device)
        p = batch["p"].to(device)
        coords_list = [c.to(device) for c in batch["coords_list"]]
        atom_mask_list = [m.to(device) for m in batch["atom_mask_list"]]
        iso_acceptor_index = batch["iso_acceptor_index"].to(device)
        iso_types = batch["iso_acceptor_type"]

        B, Lmax, A, _ = coords_list[0].shape
        Ns = 1
        x0 = torch.randn(B, Ns, Lmax, A, 3, device=device) * 5.0
        t = torch.zeros(B, Ns, device=device)
        sigma = sched.sigma(t)
        x_t = x0
        v_pred = model(
            seqs=seqs,
            aa_ids=aa_ids,
            token_mask=token_mask,
            k=k,
            p=p,
            x_t=x_t,
            sigma=sigma,
            t_scalar=t,
        )
        x1_pred = x_t + (1.0 - t[:, :, None, None, None]) * v_pred
        if hard_iso:
            x1_pred = project_iso_bb(x1_pred, iso_acceptor_index, token_mask, atom_mask_list[0])
        for i in range(B):
            name = names[i]
            seq = seqs[i]
            L = int(token_mask[i].sum().item())
            atom_mask = atom_mask_list[0][i, :L, :].cpu()
            coords = x1_pred[i, 0, :L, :, :].detach().cpu()
            out_dir = export_dir / name
            _ensure_dir(out_dir)
            pred_path = out_dir / f"{name}_pred.pdb"
            _write_backbone_pdb(
                pred_path,
                seq,
                coords,
                atom_mask,
                iso_acceptor_index=int(iso_acceptor_index[i].item()),
                iso_acceptor_type=iso_types[i],
            )
            if structure_dir:
                src_dir = Path(structure_dir) / name
                for idx in (1, 2, 3):
                    src = src_dir / f"min{idx}.pdb"
                    if src.exists():
                        shutil.copyfile(src, out_dir / f"gt_min{idx}.pdb")
                relax_src = None
                for r in (1, 2, 3):
                    cand = src_dir / f"relax{r}.pdb"
                    if cand.exists():
                        relax_src = cand
                        break
                if relax_src is not None:
                    shutil.copyfile(relax_src, out_dir / "gt_relax.pdb")
            exported += 1
            if exported >= n_export:
                break


def _check_cuda_or_fail():
    if torch.cuda.is_available():
        return
    dev_nodes = sorted(glob.glob("/dev/nvidia*"))
    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": False,
        "device_count": torch.cuda.device_count(),
        "dev_nodes": dev_nodes,
    }
    raise RuntimeError("CUDA不可用，训练被终止: " + json.dumps(info, ensure_ascii=False))


def _split_indices(n: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen).tolist()
    n_trainval = int(n * 0.9)
    trainval_idx = perm[:n_trainval]
    test_idx = perm[n_trainval:]
    return trainval_idx, test_idx


def _make_folds(indices: list[int], k: int):
    if k <= 1:
        return [indices]
    n = len(indices)
    base = n // k
    rem = n % k
    folds = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        folds.append(indices[start : start + size])
        start += size
    return folds


def _make_w_link_schedule(w_link_max: float, w_link_start: float, warmup_frac: float):
    def schedule(epoch: int, epochs: int):
        if w_link_max <= 0.0:
            return 0.0
        if warmup_frac <= 0.0:
            return w_link_max
        warmup_epochs = max(1, int(round(epochs * warmup_frac)))
        if epoch < warmup_epochs:
            return 0.0
        ramp_epochs = max(1, epochs - warmup_epochs)
        progress = (epoch - warmup_epochs + 1) / ramp_epochs
        if progress < 0.0:
            progress = 0.0
        if progress > 1.0:
            progress = 1.0
        return w_link_start + (w_link_max - w_link_start) * progress

    return schedule


def _select_collate(ds):
    base = ds.dataset if hasattr(ds, "dataset") else ds
    if isinstance(base, LassoPDBDataset):
        return collate_lasso
    return collate_toy


def _build_dls(ds, train_idx, val_idx, batch_size, world_size, rank):
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    collate_fn = _select_collate(ds)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, sampler=val_sampler, collate_fn=collate_fn)
    return train_dl, val_dl, train_sampler, val_sampler


def _train_loop(model, train_dl, val_dl, train_sampler, device, epochs, sched, w, iso_cfg, plug_cfg, link_cfg, tube_cfg, hard_iso, log_fp, rank):
    for epoch in range(epochs):
        if model.w_link_schedule is not None:
            w.w_link = model.w_link_schedule(epoch, epochs)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_stats = run_epoch(
            model=model,
            dl=train_dl,
            device=device,
            mode="train",
            opt=model.opt,
            sched=sched,
            w=w,
            iso_cfg=iso_cfg,
            plug_cfg=plug_cfg,
            link_cfg=link_cfg,
            tube_cfg=tube_cfg,
            hard_iso=hard_iso,
        )
        train_stats = _reduce_totals(train_stats, device)
        val_stats = run_epoch(
            model=model,
            dl=val_dl,
            device=device,
            mode="val",
            opt=model.opt,
            sched=sched,
            w=w,
            iso_cfg=iso_cfg,
            plug_cfg=plug_cfg,
            link_cfg=link_cfg,
            tube_cfg=tube_cfg,
            hard_iso=hard_iso,
        )
        val_stats = _reduce_totals(val_stats, device)
        if rank == 0:
            _log_line(
                "epoch="
                + str(epoch)
                + " train_loss="
                + f"{train_stats['loss']:.4f}"
                + " val_loss="
                + f"{val_stats['loss']:.4f}"
                + " train_rmsd="
                + f"{train_stats['rmsd']:.3f}"
                + " val_rmsd="
                + f"{val_stats['rmsd']:.3f}"
                + " train_lddt="
                + f"{train_stats['lddt']:.3f}"
                + " val_lddt="
                + f"{val_stats['lddt']:.3f}"
                + " train_cprec="
                + f"{train_stats['cprec']:.3f}"
                + " val_cprec="
                + f"{val_stats['cprec']:.3f}"
                + " train_link_loss="
                + f"{train_stats['link_loss']:.3f}"
                + " val_link_loss="
                + f"{val_stats['link_loss']:.3f}"
                + " train_gli="
                + f"{train_stats['gli_pred']:.4f}"
                + " val_gli="
                + f"{val_stats['gli_pred']:.4f}"
                + " train_gli_gt="
                + f"{train_stats['gli_gt']:.4f}"
                + " val_gli_gt="
                + f"{val_stats['gli_gt']:.4f}"
                + " train_time="
                + f"{train_stats['time_s']:.2f}s"
                + " val_time="
                + f"{val_stats['time_s']:.2f}s",
                log_fp,
            )
    return train_stats, val_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="pairformer")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--structure_dir", type=str, default=None)
    parser.add_argument("--split_by_seqid", action="store_true")
    parser.add_argument("--seqid_cutoff", type=float, default=0.25)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--w_link", type=float, default=None)
    parser.add_argument("--w_link_max", type=float, default=1.0)
    parser.add_argument("--w_link_start", type=float, default=0.1)
    parser.add_argument("--w_link_warmup_frac", type=float, default=0.2)
    parser.add_argument("--w_tube", type=float, default=0.05)
    parser.add_argument("--hard_iso", action="store_true")
    parser.add_argument("--cv_folds", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--results_path", type=str, default="results.txt")
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--export_val_pdb", action="store_true")
    parser.add_argument("--export_val_dir", type=str, default="val")
    parser.add_argument("--export_val_n", type=int, default=10)
    args = parser.parse_args()

    _setup_distributed()
    rank, world_size, local_rank = _dist_info()
    try:
        if args.require_cuda:
            _check_cuda_or_fail()
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        torch.manual_seed(args.seed + rank)

        mode = args.mode
        if args.structure_dir:
            ds = LassoPDBDataset(args.structure_dir)
            if args.split_by_seqid:
                split = split_by_sequence_identity(
                    fasta_root=args.structure_dir,
                    cutoff=args.seqid_cutoff,
                    test_fraction=args.test_fraction,
                    val_fraction=args.val_fraction,
                    seed=args.seed,
                )
                name_to_idx = {s.name: i for i, s in enumerate(ds.samples)}
                train_idx = [name_to_idx[n] for n in split["train"] if n in name_to_idx]
                val_idx = [name_to_idx[n] for n in split["val"] if n in name_to_idx]
                test_idx = [name_to_idx[n] for n in split["test"] if n in name_to_idx]
                trainval_idx = train_idx + val_idx
            else:
                trainval_idx, test_idx = _split_indices(len(ds), seed=args.seed)
        else:
            ds = ToyLassoDataset(n=args.n, L=args.L, K=args.K)
            trainval_idx, test_idx = _split_indices(len(ds), seed=args.seed)
        folds = _make_folds(trainval_idx, args.cv_folds)

        log_root = Path(args.log_dir) / _timestamp()
        if rank == 0:
            _ensure_dir(log_root)
        if world_size > 1:
            dist.barrier()

        sched = SigmaSchedule()
        w_link_max = args.w_link_max if args.w_link is None else args.w_link
        w_link_start = min(args.w_link_start, w_link_max)
        w = LossWeights(w_link=0.0, w_tube=args.w_tube)
        iso_cfg = IsoCfg(dmin=2.0, dmax=3.0)
        plug_cfg = PlugCfg()
        link_cfg = LinkCfg()
        tube_cfg = TubeCfg()
        w_link_schedule = _make_w_link_schedule(w_link_max, w_link_start, args.w_link_warmup_frac)

        if args.cv_folds > 1:
            for fold_id, val_idx in enumerate(folds):
                train_idx = [i for j, f in enumerate(folds) if j != fold_id for i in f]
                train_dl, val_dl, train_sampler, _ = _build_dls(ds, train_idx, val_idx, args.batch_size, world_size, rank)
                model = LassoDiffModel(mode=mode).to(device)
                if world_size > 1:
                    model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
                model.opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
                model.w_link_schedule = w_link_schedule
                log_fp = (log_root / f"fold_{fold_id}.log") if rank == 0 else None
                if rank == 0:
                    _log_line(f"fold={fold_id} train={len(train_idx)} val={len(val_idx)}", log_fp)
                _train_loop(
                    model=model,
                    train_dl=train_dl,
                    val_dl=val_dl,
                    train_sampler=train_sampler,
                    device=device,
                    epochs=args.epochs,
                    sched=sched,
                    w=w,
                    iso_cfg=iso_cfg,
                    plug_cfg=plug_cfg,
                    link_cfg=link_cfg,
                    tube_cfg=tube_cfg,
                    hard_iso=args.hard_iso,
                    log_fp=log_fp,
                    rank=rank,
                )
                if world_size > 1:
                    dist.barrier()

        train_dl, val_dl, train_sampler, val_sampler = _build_dls(ds, trainval_idx, test_idx, args.batch_size, world_size, rank)
        model = LassoDiffModel(mode=mode).to(device)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
        model.opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        model.w_link_schedule = w_link_schedule
        log_fp = (log_root / "train_full.log") if rank == 0 else None
        if rank == 0:
            _log_line(f"train_full={len(trainval_idx)} test={len(test_idx)}", log_fp)
        _train_loop(
            model=model,
            train_dl=train_dl,
            val_dl=val_dl,
            train_sampler=train_sampler,
            device=device,
            epochs=args.epochs,
            sched=sched,
            w=w,
            iso_cfg=iso_cfg,
            plug_cfg=plug_cfg,
            link_cfg=link_cfg,
            tube_cfg=tube_cfg,
            hard_iso=args.hard_iso,
            log_fp=log_fp,
            rank=rank,
        )

        test_stats = run_epoch(
            model=model,
            dl=val_dl,
            device=device,
            mode="test",
            opt=model.opt,
            sched=sched,
            w=w,
            iso_cfg=iso_cfg,
            plug_cfg=plug_cfg,
            link_cfg=link_cfg,
            tube_cfg=tube_cfg,
            hard_iso=args.hard_iso,
        )
        test_stats = _reduce_totals(test_stats, device)
        if rank == 0:
            results_path = Path(args.results_path)
            result_line = (
                "test_loss="
                + f"{test_stats['loss']:.4f}"
                + " test_rmsd="
                + f"{test_stats['rmsd']:.3f}"
                + " test_lddt="
                + f"{test_stats['lddt']:.3f}"
                + " test_cprec="
                + f"{test_stats['cprec']:.3f}"
                + " test_link_loss="
                + f"{test_stats['link_loss']:.3f}"
                + " test_gli="
                + f"{test_stats['gli_pred']:.4f}"
                + " test_gli_gt="
                + f"{test_stats['gli_gt']:.4f}"
                + " test_time="
                + f"{test_stats['time_s']:.2f}s"
            )
            results_path.write_text(result_line + "\n", encoding="utf-8")
            _log_line(result_line, log_fp)
        if world_size > 1:
            _dist_barrier(local_rank, device)
        if args.export_val_pdb and rank == 0:
            export_dir = Path(args.export_val_dir)
            if not export_dir.is_absolute():
                export_dir = log_root / export_dir
            _export_val_pdbs(
                model=model,
                ds=ds,
                val_idx=test_idx,
                device=device,
                sched=sched,
                hard_iso=args.hard_iso,
                export_dir=export_dir,
                n_export=args.export_val_n,
                structure_dir=args.structure_dir,
                seed=args.seed,
            )
        if world_size > 1:
            _dist_barrier(local_rank, device)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
