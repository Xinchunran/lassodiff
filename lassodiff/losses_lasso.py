from __future__ import annotations

from dataclasses import dataclass, field
import os
import torch
import torch.nn.functional as F


ATOM_N = 0
ATOM_CA = 1
ATOM_C = 2
ATOM_O = 3
ATOM_CISO = 4
ATOM_O1 = 5
ATOM_O2 = 6


def hinge_band(d, dmin, dmax):
    return F.relu(d - dmax) ** 2 + F.relu(dmin - d) ** 2


@dataclass
class LossWeights:
    w_flow: float = 1.0
    w_bb: float = 0.2
    w_bond: float = 1.0
    w_iso: float = 10.0
    w_plug: float = 1.0
    w_thr: float = 0.5
    w_clash: float = 0.1
    w_link: float = 0.0
    w_tube: float = 0.0
    w_angle: float = 1.0
    w_dihedral: float = 1.0


@dataclass
class IsoCfg:
    dmin: float = 1.25
    dmax: float = 1.45
    angle_min: float = 110.0
    angle_max: float = 130.0
    angle_weight: float = 1.0
    plane_weight: float = 0.1


@dataclass
class PlugCfg:
    d_plug_max: float = 6.0
    d_next_max: float = 8.0
    d_tail_min: float = 10.0
    thread_margin: float = 1.0


@dataclass
class LinkCfg:
    target: float = 1.0
    eps: float = 1e-2
    chunk_size: int = 64
    huber_delta: float = 1.0


@dataclass
class TubeCfg:
    r_tube: float = 4.0
    allow_plug_span: int = 2


@dataclass
class ClashCfg:
    radius_table: dict = field(
        default_factory=lambda: {
            "C": 1.7,
            "N": 1.55,
            "O": 1.52,
            "S": 1.8,
            "P": 1.8,
            "Mg": 1.2,
        }
    )
    alpha_default: float = 0.9
    alpha_pair_overrides: dict = field(default_factory=lambda: {("Mg", "O"): 0.7, ("Mg", "N"): 0.7})
    exclude_1_2: bool = True
    exclude_1_3: bool = True
    scale_1_4: float = 0.2
    use_softplus: bool = False
    beta: float = 10.0


def region_weights(L: int, k: int, p: int, device: torch.device, w_ring=3.0, w_loop=1.0, w_tail=1.5):
    idx = torch.arange(L, device=device)
    w = torch.ones(L, device=device) * w_loop
    w = torch.where(idx <= k, torch.tensor(w_ring, device=device), w)
    w = torch.where(idx > p, torch.tensor(w_tail, device=device), w)
    return w


def _masked_mse_per_coord(x_pred, x_true, token_mask, w_res=None, atom_mask=None, eps: float = 1e-8):
    if x_pred.dim() == 5:
        B, Ns, L, A, _C = x_pred.shape
        m = token_mask[:, None, :, None].float()
        if atom_mask is None:
            am = torch.ones((B, L, A), device=x_pred.device, dtype=torch.float32)
        else:
            am = atom_mask.float()
        m = (m * am[:, None, :, :]).expand(B, Ns, L, A)
        if w_res is not None:
            w = w_res
            if w.dim() == 1:
                w = w[None, :].expand(B, L)
            m = m * w[:, None, :, None].float()
        diff2 = (x_pred - x_true).pow(2).sum(dim=-1)
        denom = m.sum()
        return (diff2 * m).sum() / (denom * 3.0 * Ns + eps)

    if x_pred.dim() == 4:
        B, d1, d2, _C = x_pred.shape
        if d1 == token_mask.shape[1]:
            L, A = d1, d2
            m = token_mask[:, :, None].float()
            if atom_mask is None:
                am = torch.ones((B, L, A), device=x_pred.device, dtype=torch.float32)
            else:
                am = atom_mask.float()
            m = m * am
            if w_res is not None:
                w = w_res
                if w.dim() == 1:
                    w = w[None, :].expand(B, L)
                m = m * w[:, :, None].float()
            diff2 = (x_pred - x_true).pow(2).sum(dim=-1)
            denom = m.sum()
            return (diff2 * m).sum() / (denom * 3.0 + eps)
        else:
            Ns, L = d1, d2
            m = token_mask[:, None, :].float().expand(B, Ns, L)
            if atom_mask is not None and atom_mask.dim() == 2:
                m = m * atom_mask[:, None, :].float()
            if w_res is not None:
                w = w_res
                if w.dim() == 1:
                    w = w[None, :].expand(B, L)
                m = m * w[:, None, :].float()
            diff2 = (x_pred - x_true).pow(2).sum(dim=-1)
            denom = m.sum()
            return (diff2 * m).sum() / (denom * 3.0 + eps)

    if x_pred.dim() == 3:
        B, L, _C = x_pred.shape
        m = token_mask.float()
        if atom_mask is not None and atom_mask.dim() == 2:
            m = m * atom_mask.float()
        if w_res is not None:
            w = w_res
            if w.dim() == 1:
                w = w[None, :].expand(B, L)
            m = m * w.float()
        diff2 = (x_pred - x_true).pow(2).sum(dim=-1)
        denom = m.sum()
        return (diff2 * m).sum() / (denom * 3.0 + eps)

    raise ValueError(f"Unsupported tensor shape for masked MSE: {tuple(x_pred.shape)}")


def _weighted_rigid_align(
    x: torch.Tensor,
    x_target: torch.Tensor,
    atom_weight: torch.Tensor,
    stop_gradient: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    if stop_gradient:
        with torch.no_grad():
            return _weighted_rigid_align(x, x_target, atom_weight, stop_gradient=False, eps=eps).detach()

    w = atom_weight.float()
    w_sum = w.sum(dim=-1, keepdim=True).clamp(min=eps)
    x_centroid = (x * w[..., None]).sum(dim=-2, keepdim=True) / w_sum[..., None]
    y_centroid = (x_target * w[..., None]).sum(dim=-2, keepdim=True) / w_sum[..., None]
    x0 = x - x_centroid
    y0 = x_target - y_centroid
    h = torch.matmul((x0 * w[..., None]).transpose(-2, -1), y0)
    u, _s, vh = torch.linalg.svd(h)
    v = vh.transpose(-2, -1)
    ut = u.transpose(-2, -1)
    det = torch.linalg.det(torch.matmul(v, ut))
    diag = torch.stack([torch.ones_like(det), torch.ones_like(det), det], dim=-1)
    r = torch.matmul(v, torch.matmul(torch.diag_embed(diag).to(v.device), ut))
    x_aligned = torch.matmul(x0, r.transpose(-2, -1)) + y_centroid
    return x_aligned


def _align_true_to_pred(
    x_pred: torch.Tensor,
    x_true: torch.Tensor,
    token_mask: torch.Tensor,
    w_res: torch.Tensor | None = None,
    atom_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if x_pred.dim() != 5:
        raise ValueError(f"Expected x_pred dim=5, got shape {tuple(x_pred.shape)}")
    B, Ns, L, A, _ = x_pred.shape
    if atom_mask is None:
        am = torch.ones((B, L, A), device=x_pred.device, dtype=torch.float32)
    else:
        am = atom_mask.float()
    w = token_mask[:, None, :, None].float() * am[:, None, :, :]
    if w_res is not None:
        w_local = w_res
        if w_local.dim() == 1:
            w_local = w_local[None, :].expand(B, L)
        w = w * w_local[:, None, :, None].float()
    x_pred_f = x_pred.reshape(B, Ns, L * A, 3)
    x_true_f = x_true.reshape(B, Ns, L * A, 3)
    w_f = w.reshape(B, Ns, L * A)
    x_true_aligned = _weighted_rigid_align(x_true_f, x_pred_f, w_f, stop_gradient=True)
    return x_true_aligned.reshape(B, Ns, L, A, 3)


def loss_ca_adj(x1_pred, x1_true, token_mask, w_res=None, atom_mask=None, eps: float = 1e-8):
    if x1_pred.dim() != 5:
        raise ValueError(f"Expected x1_pred dim=5, got shape {tuple(x1_pred.shape)}")
    B, Ns, L, _A, _C = x1_pred.shape
    if L < 2:
        return x1_pred.new_tensor(0.0)
    if atom_mask is None:
        ca_ok = token_mask
    else:
        ca_ok = token_mask & atom_mask[:, :, ATOM_CA].bool()
    m = (ca_ok[:, None, :-1] & ca_ok[:, None, 1:]).float()
    if w_res is not None:
        w = w_res
        if w.dim() == 1:
            w = w[None, :].expand(B, L)
        w_pair = 0.5 * (w[:, :-1] + w[:, 1:])
        m = m * w_pair[:, None, :].float()
    x_pred_ca = x1_pred[..., ATOM_CA, :]
    x_true_ca = x1_true[..., ATOM_CA, :]
    d_pred = torch.linalg.vector_norm(x_pred_ca[:, :, 1:, :] - x_pred_ca[:, :, :-1, :], dim=-1)
    d_true = torch.linalg.vector_norm(x_true_ca[:, :, 1:, :] - x_true_ca[:, :, :-1, :], dim=-1)
    diff2 = (d_pred - d_true).pow(2)
    denom = m.sum().clamp(min=1.0)
    return (diff2 * m).sum() / (denom + eps)


def loss_smooth_lddt_ca(
    x_pred_ca: torch.Tensor,
    x_true_ca: torch.Tensor,
    token_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if x_pred_ca.dim() != 4 or x_true_ca.dim() != 4:
        raise ValueError(
            f"Expected x_pred_ca/x_true_ca shape [B,Ns,L,3], got {tuple(x_pred_ca.shape)} / {tuple(x_true_ca.shape)}"
        )
    B, Ns, L, _C = x_pred_ca.shape
    if L < 2:
        return x_pred_ca.new_tensor(0.0)

    mask = token_mask.float()
    xi = x_pred_ca[:, :, :, None, :]
    xj = x_pred_ca[:, :, None, :, :]
    yi = x_true_ca[:, :, :, None, :]
    yj = x_true_ca[:, :, None, :, :]
    d_pred = torch.linalg.vector_norm(xi - xj, dim=-1)
    d_true = torch.linalg.vector_norm(yi - yj, dim=-1)

    eye = torch.eye(L, device=x_pred_ca.device).bool()
    valid = (mask[:, None, :, None] * mask[:, None, None, :]).bool()
    valid = valid & (~eye[None, None, :, :])
    valid = valid & (d_true < cutoff)

    diff = torch.abs(d_pred - d_true)
    score = (
        (torch.sigmoid(0.5 - diff) + torch.sigmoid(1.0 - diff) + torch.sigmoid(2.0 - diff) + torch.sigmoid(4.0 - diff))
        / 4.0
    )
    num = (score * valid.float()).sum()
    denom = valid.float().sum().clamp(min=1.0)
    lddt = num / (denom + eps)
    return 1.0 - lddt


def _frame_from_n_ca_c(x_bb: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    n = x_bb[..., ATOM_N, :]
    ca = x_bb[..., ATOM_CA, :]
    c = x_bb[..., ATOM_C, :]
    e1 = c - ca
    e1 = e1 / (torch.linalg.vector_norm(e1, dim=-1, keepdim=True) + eps)
    u2 = n - ca
    u2 = u2 - (u2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = u2 / (torch.linalg.vector_norm(u2, dim=-1, keepdim=True) + eps)
    e3 = torch.linalg.cross(e1, e2, dim=-1)
    e3 = e3 / (torch.linalg.vector_norm(e3, dim=-1, keepdim=True) + eps)
    R = torch.stack([e1, e2, e3], dim=-1)
    t = ca
    return R, t


def loss_fape_ca(
    x1_pred: torch.Tensor,
    x1_true: torch.Tensor,
    token_mask: torch.Tensor,
    w_res: torch.Tensor | None = None,
    atom_mask: torch.Tensor | None = None,
    clamp: float = 10.0,
    chunk: int = 64,
    eps: float = 1e-8,
) -> torch.Tensor:
    if x1_pred.dim() != 5 or x1_true.dim() != 5:
        raise ValueError(f"Expected x1_pred/x1_true [B,Ns,L,A,3], got {tuple(x1_pred.shape)} / {tuple(x1_true.shape)}")
    B, Ns, L, A, _C = x1_pred.shape
    if L < 2:
        return x1_pred.new_tensor(0.0)
    if A <= ATOM_C:
        return x1_pred.new_tensor(0.0)

    if atom_mask is None:
        am = torch.ones((B, L, A), device=x1_pred.device, dtype=torch.bool)
    else:
        am = atom_mask.bool()
    frame_ok = token_mask & am[:, :, ATOM_N] & am[:, :, ATOM_CA] & am[:, :, ATOM_C]
    pt_ok = token_mask & am[:, :, ATOM_CA]

    R_pred, t_pred = _frame_from_n_ca_c(x1_pred[..., :4, :], eps=eps)
    R_true, t_true = _frame_from_n_ca_c(x1_true[..., :4, :], eps=eps)

    p_pred = x1_pred[..., ATOM_CA, :]
    p_true = x1_true[..., ATOM_CA, :]

    total = x1_pred.new_tensor(0.0)
    denom = x1_pred.new_tensor(0.0)
    wj = pt_ok[:, None, :].float()
    if w_res is not None:
        w = w_res
        if w.dim() == 1:
            w = w[None, :].expand(B, L)
        wj = wj * w[:, None, :].float()

    for i0 in range(0, L, int(chunk)):
        i1 = min(L, i0 + int(chunk))
        ok_i = frame_ok[:, i0:i1].float()
        if w_res is not None:
            ok_i = ok_i * w[:, i0:i1].float()
        ok_i = ok_i[:, None, :, None]

        tp = t_pred[:, :, i0:i1, :]
        tt = t_true[:, :, i0:i1, :]
        Rp = R_pred[:, :, i0:i1, :, :]
        Rt = R_true[:, :, i0:i1, :, :]

        pred_local = torch.einsum("bnilj,bnijk->bnilk", p_pred[:, :, None, :, :] - tp[:, :, :, None, :], Rp)
        true_local = torch.einsum("bnilj,bnijk->bnilk", p_true[:, :, None, :, :] - tt[:, :, :, None, :], Rt)
        diff = pred_local - true_local
        err = torch.linalg.vector_norm(diff, dim=-1)
        if float(clamp) > 0:
            err = torch.clamp(err, max=float(clamp))
        w_pair = ok_i * wj[:, :, None, :]
        total = total + (err.pow(2) * w_pair).sum()
        denom = denom + w_pair.sum()

    return total / (denom.clamp(min=1.0) + eps)


def loss_flow(v_pred, v_star, token_mask, w_res=None, atom_mask=None):
    return _masked_mse_per_coord(v_pred, v_star, token_mask, w_res=w_res, atom_mask=atom_mask)


def loss_bb(x1_pred, x1_true, token_mask, w_res=None, atom_mask=None):
    if os.environ.get("LASSODIFF_ALIGN_BB", "1") == "1" and x1_pred.dim() == 5:
        x1_true = _align_true_to_pred(x1_pred, x1_true, token_mask, w_res=w_res, atom_mask=atom_mask)
    return _masked_mse_per_coord(x1_pred, x1_true, token_mask, w_res=w_res, atom_mask=atom_mask)


def loss_bond_bb(x, token_mask, atom_mask=None):
    parts = loss_bond_bb_parts(x, token_mask, atom_mask=atom_mask)
    return parts["loss"]


def loss_bond_bb_parts(x, token_mask, atom_mask=None):
    if x.dim() == 5:
        tok = token_mask[:, None, :].bool()
        if atom_mask is None:
            am = tok[:, :, :, None].expand(-1, x.shape[1], -1, x.shape[3])
        else:
            am = atom_mask[:, None, :, :].bool()
    else:
        tok = token_mask.bool()
        if atom_mask is None:
            am = tok[:, :, None].expand(-1, -1, x.shape[2])
        else:
            am = atom_mask.bool()

    def _pair(res_a, atom_a, res_b, atom_b, target):
        pa = x[..., res_a, atom_a, :]
        pb = x[..., res_b, atom_b, :]
        d = torch.linalg.vector_norm(pa - pb, dim=-1)
        if x.dim() == 5:
            ok = tok[..., res_a] & tok[..., res_b] & am[..., res_a, atom_a] & am[..., res_b, atom_b]
        else:
            ok = tok[:, res_a] & tok[:, res_b] & am[:, res_a, atom_a] & am[:, res_b, atom_b]
        ok_f = ok.float()
        return ((d - target) ** 2) * ok_f, ok_f

    B = x.shape[0]
    L = x.shape[2] if x.dim() == 5 else x.shape[1]
    dev = x.device
    idx = torch.arange(L, device=dev)
    idx_next = idx + 1
    valid = idx_next < L
    idx = idx[valid]
    idx_next = idx_next[valid]
    if idx.numel() == 0:
        z = x.new_tensor(0.0)
        return {
            "loss": z,
            "nca": z,
            "cac": z,
            "co": z,
            "cn": z,
            "nca_pairs": z,
            "cac_pairs": z,
            "co_pairs": z,
            "cn_pairs": z,
        }

    l1, m1 = _pair(idx, ATOM_N, idx, ATOM_CA, 1.46)
    l2, m2 = _pair(idx, ATOM_CA, idx, ATOM_C, 1.52)
    l3, m3 = _pair(idx, ATOM_C, idx, ATOM_O, 1.23)
    l4, m4 = _pair(idx, ATOM_C, idx_next, ATOM_N, 1.33)
    denom1 = m1.sum().clamp(min=1.0)
    denom2 = m2.sum().clamp(min=1.0)
    denom3 = m3.sum().clamp(min=1.0)
    denom4 = m4.sum().clamp(min=1.0)

    p1 = l1.sum() / denom1
    p2 = l2.sum() / denom2
    p3 = l3.sum() / denom3
    p4 = l4.sum() / denom4

    loss = (l1.sum() + l2.sum() + l3.sum() + l4.sum()) / (m1.sum() + m2.sum() + m3.sum() + m4.sum()).clamp(min=1.0)

    return {
        "loss": loss,
        "nca": p1,
        "cac": p2,
        "co": p3,
        "cn": p4,
        "nca_pairs": m1.sum(),
        "cac_pairs": m2.sum(),
        "co_pairs": m3.sum(),
        "cn_pairs": m4.sum(),
    }


def loss_angle_bb(x, token_mask, atom_mask=None):
    if x.dim() == 5:
        tok = token_mask[:, None, :].bool()
        if atom_mask is None:
            am = tok[:, :, :, None].expand(-1, x.shape[1], -1, x.shape[3])
        else:
            am = atom_mask[:, None, :, :].bool()
    else:
        tok = token_mask.bool()
        if atom_mask is None:
            am = tok[:, :, None].expand(-1, -1, x.shape[2])
        else:
            am = atom_mask.bool()

    def _trip(res_a, atom_a, res_b, atom_b, res_c, atom_c, target_deg):
        pa = x[..., res_a, atom_a, :]
        pb = x[..., res_b, atom_b, :]
        pc = x[..., res_c, atom_c, :]

        # Use cosine similarity directly for stability and bounded loss
        # target_deg is in degrees. Convert to radians.
        target_rad = torch.tensor(target_deg * torch.pi / 180.0, device=dev)

        # Calculate angle (0 to pi)
        # We can implement _angle_rad or just use _angle * pi / 180
        # But let's reuse _angle for now and convert
        deg = _angle(pa, pb, pc)
        rad = deg * torch.pi / 180.0

        # Loss: 1 - cos(rad - target_rad)
        loss_val = 1.0 - torch.cos(rad - target_rad)

        if x.dim() == 5:
            ok = (
                tok[..., res_a]
                & tok[..., res_b]
                & tok[..., res_c]
                & am[..., res_a, atom_a]
                & am[..., res_b, atom_b]
                & am[..., res_c, atom_c]
            )
        else:
            ok = (
                tok[:, res_a]
                & tok[:, res_b]
                & tok[:, res_c]
                & am[:, res_a, atom_a]
                & am[:, res_b, atom_b]
                & am[:, res_c, atom_c]
            )
        ok_f = ok.float()
        return loss_val * ok_f, ok_f

    B = x.shape[0]
    L = x.shape[2] if x.dim() == 5 else x.shape[1]
    dev = x.device
    idx = torch.arange(L, device=dev)
    idx_next = idx + 1
    valid = idx_next < L
    idx = idx[valid]
    idx_next = idx_next[valid]
    if idx.numel() == 0:
        return x.new_tensor(0.0)

    # N-CA-C: 111.2
    l1, m1 = _trip(idx, ATOM_N, idx, ATOM_CA, idx, ATOM_C, 111.2)
    # CA-C-N: 116.2
    l2, m2 = _trip(idx, ATOM_CA, idx, ATOM_C, idx_next, ATOM_N, 116.2)
    # C-N-CA: 121.7
    l3, m3 = _trip(idx, ATOM_C, idx_next, ATOM_N, idx_next, ATOM_CA, 121.7)
    # CA-C-O: 120.8 (Planar carbonyl)
    l4, m4 = _trip(idx, ATOM_CA, idx, ATOM_C, idx, ATOM_O, 120.8)

    loss = l1.sum() + l2.sum() + l3.sum() + l4.sum()
    denom = (m1.sum() + m2.sum() + m3.sum() + m4.sum()).clamp(min=1.0)
    return loss / denom


def _dihedral(a, b, c, d, eps=1e-8):
    b0 = -1.0 * (b - a)
    b1 = c - b
    b2 = d - c

    b1 = b1 / (torch.linalg.vector_norm(b1, dim=-1, keepdim=True) + eps)

    v = b0 - (b0 * b1).sum(dim=-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(dim=-1, keepdim=True) * b1

    x = (v * w).sum(dim=-1)
    y = (torch.linalg.cross(b1, v) * w).sum(dim=-1)

    return torch.atan2(y, x)


def loss_dihedral_bb(x, token_mask, atom_mask=None):
    if x.dim() == 5:
        tok = token_mask[:, None, :].bool()
        if atom_mask is None:
            am = tok[:, :, :, None].expand(-1, x.shape[1], -1, x.shape[3])
        else:
            am = atom_mask[:, None, :, :].bool()
    else:
        tok = token_mask.bool()
        if atom_mask is None:
            am = tok[:, :, None].expand(-1, -1, x.shape[2])
        else:
            am = atom_mask.bool()

    def _quad(res_a, atom_a, res_b, atom_b, res_c, atom_c, res_d, atom_d, target_rad):
        pa = x[..., res_a, atom_a, :]
        pb = x[..., res_b, atom_b, :]
        pc = x[..., res_c, atom_c, :]
        pd = x[..., res_d, atom_d, :]

        torsion = _dihedral(pa, pb, pc, pd)
        loss_val = 1.0 - torch.cos(torsion - target_rad)

        if x.dim() == 5:
            ok = (
                tok[..., res_a]
                & tok[..., res_b]
                & tok[..., res_c]
                & tok[..., res_d]
                & am[..., res_a, atom_a]
                & am[..., res_b, atom_b]
                & am[..., res_c, atom_c]
                & am[..., res_d, atom_d]
            )
        else:
            ok = (
                tok[:, res_a]
                & tok[:, res_b]
                & tok[:, res_c]
                & tok[:, res_d]
                & am[:, res_a, atom_a]
                & am[:, res_b, atom_b]
                & am[:, res_c, atom_c]
                & am[:, res_d, atom_d]
            )
        ok_f = ok.float()
        return loss_val * ok_f, ok_f

    B = x.shape[0]
    L = x.shape[2] if x.dim() == 5 else x.shape[1]
    dev = x.device
    idx = torch.arange(L, device=dev)
    idx_next = idx + 1
    valid = idx_next < L
    idx = idx[valid]
    idx_next = idx_next[valid]
    if idx.numel() == 0:
        return x.new_tensor(0.0)

    # Omega: CA_i, C_i, N_{i+1}, CA_{i+1} -> 180 deg (pi rad)
    l1, m1 = _quad(idx, ATOM_CA, idx, ATOM_C, idx_next, ATOM_N, idx_next, ATOM_CA, torch.pi)

    loss = l1.sum()
    denom = m1.sum().clamp(min=1.0)
    return loss / denom


def loss_iso_ca(x1_pred, k_ring_end, token_mask, iso_cfg: IsoCfg = IsoCfg()):
    B, Ns, L, _ = x1_pred.shape
    b = torch.arange(B, device=x1_pred.device)
    k = k_ring_end.clamp(0, L - 1)
    p0 = x1_pred[:, :, 0, :]
    pk = x1_pred[b, :, k, :]
    d = torch.linalg.vector_norm(p0 - pk, dim=-1)
    ok = token_mask[:, 0] & token_mask[b, k]
    ok = ok[:, None].float()
    return (hinge_band(d, iso_cfg.dmin, iso_cfg.dmax) * ok).sum() / (ok.sum() + 1e-8)


def _angle(a, b, c, eps=1e-8):
    v1 = a - b
    v2 = c - b
    n1 = torch.linalg.vector_norm(v1, dim=-1).clamp(min=eps)
    n2 = torch.linalg.vector_norm(v2, dim=-1).clamp(min=eps)
    cos = (v1 * v2).sum(dim=-1) / (n1 * n2)
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.acos(cos) * (180.0 / torch.pi)


def loss_iso_bb(x1_pred, iso_acceptor_index, token_mask, atom_mask, iso_cfg: IsoCfg = IsoCfg(), eps: float = 1e-8):
    B, Ns, L, A, _ = x1_pred.shape
    b = torch.arange(B, device=x1_pred.device)
    idx = iso_acceptor_index.clamp(0, L - 1)
    valid = iso_acceptor_index >= 0
    n_mask = atom_mask[:, 0, ATOM_N] & token_mask[:, 0]
    c_mask = atom_mask[b, idx, ATOM_CISO] & token_mask[b, idx]
    ok = valid & n_mask & c_mask
    ok_f = ok[:, None].float()

    n_pos = x1_pred[:, :, 0, ATOM_N, :]
    c_pos = x1_pred[b, :, idx, ATOM_CISO, :]
    d = torch.linalg.vector_norm(n_pos - c_pos, dim=-1)
    L_d = hinge_band(d, iso_cfg.dmin, iso_cfg.dmax)
    d_term = (L_d * ok_f).sum() / (ok_f.sum() + eps)

    o1_mask = atom_mask[b, idx, ATOM_O1] & token_mask[b, idx]
    o2_mask = atom_mask[b, idx, ATOM_O2] & token_mask[b, idx]
    o1_pos = x1_pred[b, :, idx, ATOM_O1, :]
    o2_pos = x1_pred[b, :, idx, ATOM_O2, :]
    a1 = _angle(n_pos, c_pos, o1_pos, eps=eps)
    a2 = _angle(n_pos, c_pos, o2_pos, eps=eps)
    L_a1 = hinge_band(a1, iso_cfg.angle_min, iso_cfg.angle_max)
    L_a2 = hinge_band(a2, iso_cfg.angle_min, iso_cfg.angle_max)
    m1 = (ok & o1_mask)[:, None].float()
    m2 = (ok & o2_mask)[:, None].float()
    both = (ok & o1_mask & o2_mask)[:, None].float()
    one = (ok & (o1_mask | o2_mask))[:, None].float()
    angle_min = torch.minimum(L_a1, L_a2)
    angle_pick = torch.where(both.bool(), angle_min, L_a1 * m1 + L_a2 * m2)
    angle_term = (angle_pick * one).sum() / (one.sum() + eps)

    plane_ok = (ok & o1_mask & o2_mask)[:, None].float()
    v1 = o1_pos - c_pos
    v2 = o2_pos - c_pos
    normal = torch.linalg.cross(v1, v2, dim=-1)
    normal = normal / (torch.linalg.vector_norm(normal, dim=-1, keepdim=True) + eps)
    plane_dist = (normal * (n_pos - c_pos)).sum(dim=-1).abs()
    plane_term = (plane_dist.pow(2) * plane_ok).sum() / (plane_ok.sum() + eps)

    return d_term + iso_cfg.angle_weight * angle_term + iso_cfg.plane_weight * plane_term


def project_iso_ca(x1_pred, k_ring_end, token_mask, iso_cfg: IsoCfg = IsoCfg(), eps: float = 1e-8):
    B, Ns, L, _ = x1_pred.shape
    b = torch.arange(B, device=x1_pred.device)
    k = k_ring_end.clamp(0, L - 1)
    ok = token_mask[:, 0] & token_mask[b, k]
    ok_f = ok[:, None, None].float()

    x = x1_pred.clone()
    p0 = x[:, :, 0, :]
    pk = x[b, :, k, :]

    v = pk - p0
    d = torch.linalg.vector_norm(v, dim=-1).clamp(min=eps)
    d_target = d.clamp(min=iso_cfg.dmin, max=iso_cfg.dmax)
    adjust = 0.5 * (d - d_target)[:, :, None] * (v / d[:, :, None])
    adjust = adjust * ok_f

    x[:, :, 0, :] = p0 + adjust
    x[b, :, k, :] = pk - adjust
    return x


def project_iso_bb(x1_pred, iso_acceptor_index, token_mask, atom_mask, iso_cfg: IsoCfg = IsoCfg(), eps: float = 1e-8):
    B, Ns, L, A, _ = x1_pred.shape
    b = torch.arange(B, device=x1_pred.device)
    idx = iso_acceptor_index.clamp(0, L - 1)
    valid = iso_acceptor_index >= 0
    ok = (
        valid
        & token_mask[:, 0]
        & token_mask[b, idx]
        & atom_mask[:, 0, ATOM_N]
        & atom_mask[b, idx, ATOM_CISO]
    )
    ok_f = ok[:, None, None].float()

    x = x1_pred.clone()
    n_pos = x[:, :, 0, ATOM_N, :]
    c_pos = x[b, :, idx, ATOM_CISO, :]

    v = c_pos - n_pos
    d = torch.linalg.vector_norm(v, dim=-1).clamp(min=eps)
    d_target = d.clamp(min=iso_cfg.dmin, max=iso_cfg.dmax)
    adjust = 0.5 * (d - d_target)[:, :, None] * (v / d[:, :, None])
    adjust = adjust * ok_f

    x[:, :, 0, ATOM_N, :] = n_pos + adjust
    x[b, :, idx, ATOM_CISO, :] = c_pos - adjust
    return x


def gauss_linking_integral_ca(x1_pred, k_ring_end, p_plug, token_mask, link_cfg: LinkCfg = LinkCfg()):
    B, Ns, L, _ = x1_pred.shape
    dev = x1_pred.device
    b = torch.arange(B, device=dev)

    k = k_ring_end.clamp(0, L - 1)
    p = p_plug.clamp(0, L - 1)
    idx = torch.arange(L, device=dev)
    ring_res_mask = (idx[None, :] <= k[:, None]) & token_mask
    tail_start = torch.maximum(p + 1, k + 1).clamp(0, L - 1)
    tail_res_mask = (idx[None, :] >= tail_start[:, None]) & token_mask

    x0 = x1_pred[:, :, :-1, :]
    x1 = x1_pred[:, :, 1:, :]
    seg_mid = 0.5 * (x0 + x1)
    seg_d = x1 - x0

    ring_seg_mask = (ring_res_mask[:, :-1] & ring_res_mask[:, 1:]).float()[:, None, :]
    tail_seg_mask = (tail_res_mask[:, :-1] & tail_res_mask[:, 1:]).float()[:, None, :]

    rk = x1_pred[b, :, k, :]
    r0 = x1_pred[:, :, 0, :]
    closure_mid = 0.5 * (rk + r0)[:, :, None, :]
    closure_d = (r0 - rk)[:, :, None, :]
    closure_mask = (token_mask[:, 0] & token_mask[b, k]).float()[:, None, None]

    ring_mid = torch.cat([seg_mid, closure_mid], dim=2)
    ring_d = torch.cat([seg_d, closure_d], dim=2)
    ring_mask = torch.cat([ring_seg_mask, closure_mask], dim=2)

    tail_mid = seg_mid
    tail_d = seg_d
    tail_mask = tail_seg_mask

    total = torch.zeros((B, Ns), device=dev, dtype=x1_pred.dtype)
    denom = 4.0 * torch.pi
    chunk = max(int(link_cfg.chunk_size), 1)
    eps = float(link_cfg.eps)

    for j0 in range(0, L - 1, chunk):
        j1 = min(L - 1, j0 + chunk)
        tm = tail_mid[:, :, j0:j1, :]
        td = tail_d[:, :, j0:j1, :]
        tmask = tail_mask[:, :, j0:j1]

        diff = ring_mid[:, :, :, None, :] - tm[:, :, None, :, :]
        cross = torch.linalg.cross(ring_d[:, :, :, None, :], td[:, :, None, :, :], dim=-1)
        num = (cross * diff).sum(dim=-1)
        r3 = torch.linalg.vector_norm(diff, dim=-1).clamp(min=eps).pow(3)
        contrib = (num / r3) * ring_mask[:, :, :, None] * tmask[:, :, None, :]
        total = total + contrib.sum(dim=(-1, -2))

    return total / denom


def loss_link_ca(x1_pred, k_ring_end, p_plug, token_mask, gli_gt=None, link_cfg: LinkCfg = LinkCfg()):
    gli = gauss_linking_integral_ca(x1_pred, k_ring_end, p_plug, token_mask, link_cfg=link_cfg)
    target = gli_gt if gli_gt is not None else float(link_cfg.target)
    target = torch.as_tensor(target, device=gli.device, dtype=gli.dtype)
    if target.shape != gli.shape:
        if target.numel() == 1:
            target = target.expand_as(gli)
        elif target.dim() == 1 and target.shape[0] == gli.shape[0]:
            target = target[:, None].expand_as(gli)
        else:
            target = target.expand_as(gli)
    return F.smooth_l1_loss(gli, target, beta=float(link_cfg.huber_delta))


def loss_ring_tube_ca(x1_pred, k_ring_end, p_plug, token_mask, tube_cfg: TubeCfg = TubeCfg(), eps: float = 1e-8):
    B, Ns, L, _ = x1_pred.shape
    dev = x1_pred.device
    idx = torch.arange(L, device=dev)
    k = k_ring_end.clamp(0, L - 1)
    p = p_plug.clamp(0, L - 1)

    ring_mask = ((idx[None, :] <= k[:, None]) & token_mask).float()[:, None, :]
    allow = (p + int(tube_cfg.allow_plug_span)).clamp(0, L - 1)
    tail_mask = ((idx[None, :] >= allow[:, None]) & token_mask).float()[:, None, :]

    total = torch.zeros((), device=dev, dtype=x1_pred.dtype)
    count = torch.zeros((), device=dev, dtype=x1_pred.dtype)
    chunk = 64

    for j0 in range(0, L, chunk):
        j1 = min(L, j0 + chunk)
        tmask = tail_mask[:, :, j0:j1]
        if float(tmask.sum().item()) == 0.0:
            continue
        tail_pts = x1_pred[:, :, j0:j1, :]
        diff = tail_pts[:, :, :, None, :] - x1_pred[:, :, None, :, :]
        d = torch.linalg.vector_norm(diff, dim=-1)
        d = d + (1.0 - ring_mask[:, :, None, :]) * 1e8
        min_d = d.min(dim=-1).values
        pen = F.relu(float(tube_cfg.r_tube) - min_d).pow(2) * tmask
        total = total + pen.sum()
        count = count + tmask.sum()

    return total / (count + eps)


def loss_plug_thread_ca(x1_pred, k_ring_end, p_plug, token_mask, plug_cfg: PlugCfg = PlugCfg()):
    B, Ns, L, _ = x1_pred.shape
    dev = x1_pred.device
    b = torch.arange(B, device=dev)

    k = k_ring_end.clamp(0, L - 1)
    p = p_plug.clamp(0, L - 1)

    a1 = torch.ones(B, dtype=torch.long, device=dev) * min(1, L - 1)
    a2 = (k // 2).clamp(0, L - 1)
    a3 = (k - 1).clamp(0, L - 1)
    t2 = (p + 2).clamp(0, L - 1)
    tail = torch.ones(B, dtype=torch.long, device=dev) * (L - 1)

    def gather(res_idx):
        return x1_pred[b, :, res_idx, :]

    xp = gather(p)
    xn = gather(t2)
    xt = gather(tail)

    xa1 = gather(a1)
    xa2 = gather(a2)
    xa3 = gather(a3)

    d_plug = torch.stack(
        [
            torch.linalg.vector_norm(xp - xa1, dim=-1),
            torch.linalg.vector_norm(xp - xa2, dim=-1),
            torch.linalg.vector_norm(xp - xa3, dim=-1),
        ],
        dim=-1,
    )
    L_plug = F.relu(d_plug - plug_cfg.d_plug_max).pow(2).mean(dim=-1)

    d_next = torch.stack(
        [
            torch.linalg.vector_norm(xn - xa1, dim=-1),
            torch.linalg.vector_norm(xn - xa2, dim=-1),
            torch.linalg.vector_norm(xn - xa3, dim=-1),
        ],
        dim=-1,
    )
    L_next = F.relu(d_next - plug_cfg.d_next_max).pow(2).mean(dim=-1)

    d_tail = torch.linalg.vector_norm(xt - xa2, dim=-1)
    L_tail = F.relu(plug_cfg.d_tail_min - d_tail).pow(2)

    v1 = xa2 - xa1
    v2 = xa3 - xa1
    normal = torch.linalg.cross(v1, v2)
    normal = normal / (torch.linalg.vector_norm(normal, dim=-1, keepdim=True) + 1e-8)
    s_p = ((xp - xa1) * normal).sum(dim=-1)
    s_t = ((xt - xa1) * normal).sum(dim=-1)
    prod = s_p * s_t
    L_thr = F.relu(prod + plug_cfg.thread_margin).pow(2)

    ok = token_mask[:, 0] & token_mask[b, k] & token_mask[b, p]
    ok = ok[:, None].float()

    return ((L_plug + L_next + L_tail) * ok).sum() / (ok.sum() + 1e-8), (L_thr * ok).sum() / (ok.sum() + 1e-8)


def loss_clash_ca(x1_pred, token_mask, r_min=3.6):
    B, Ns, L, _ = x1_pred.shape
    mask = token_mask.float()
    xi = x1_pred[:, :, :, None, :]
    xj = x1_pred[:, :, None, :, :]
    d = torch.linalg.vector_norm(xi - xj, dim=-1)

    eye = torch.eye(L, device=x1_pred.device).bool()
    d = d.masked_fill(eye[None, None, :, :], 1e8)

    ok = mask[:, None, :, None] * mask[:, None, None, :]
    pen = F.relu(r_min - d).pow(2) * ok
    return pen.sum() / (ok.sum() + 1e-8)


def loss_clash_bb(
    x1_pred,
    token_mask,
    atom_mask,
    iso_acceptor_index,
    cfg: ClashCfg = ClashCfg(),
    eps: float = 1e-8,
):
    B, Ns, L, A, _ = x1_pred.shape
    dev = x1_pred.device
    atom_elements = ["N", "C", "C", "O", "C", "O", "O"]
    if A > len(atom_elements):
        atom_elements = atom_elements + ["C"] * (A - len(atom_elements))
    atom_elements = atom_elements[:A]
    radii = torch.tensor([float(cfg.radius_table.get(e, 1.7)) for e in atom_elements], device=dev)
    alpha = torch.full((A, A), float(cfg.alpha_default), device=dev)
    for (e1, e2), a in cfg.alpha_pair_overrides.items():
        if e1 in atom_elements and e2 in atom_elements:
            i = atom_elements.index(e1)
            j = atom_elements.index(e2)
            alpha[i, j] = float(a)
            alpha[j, i] = float(a)
    atom_id = torch.arange(L * A, device=dev) % A
    r_i = radii[atom_id]
    r_j = r_i[:, None]
    alpha_pair = alpha[atom_id][:, atom_id]
    r_ij = alpha_pair * (r_i[:, None] + r_j)

    base_exclude = torch.zeros((L * A, L * A), dtype=torch.bool, device=dev)
    weight = torch.ones((L * A, L * A), device=dev)

    def idx(res, atom):
        return res * A + atom

    for i in range(L):
        if A > ATOM_CA:
            n = idx(i, ATOM_N)
            ca = idx(i, ATOM_CA)
            c = idx(i, ATOM_C)
            o = idx(i, ATOM_O)
            if cfg.exclude_1_2:
                base_exclude[n, ca] = True
                base_exclude[ca, n] = True
                base_exclude[ca, c] = True
                base_exclude[c, ca] = True
                base_exclude[c, o] = True
                base_exclude[o, c] = True
            if cfg.exclude_1_3:
                base_exclude[n, c] = True
                base_exclude[c, n] = True
                base_exclude[ca, o] = True
                base_exclude[o, ca] = True
            if cfg.scale_1_4 < 1.0:
                weight[n, o] = float(cfg.scale_1_4)
                weight[o, n] = float(cfg.scale_1_4)
        if A > ATOM_CISO:
            ciso = idx(i, ATOM_CISO)
            if A > ATOM_O1 and cfg.exclude_1_2:
                o1 = idx(i, ATOM_O1)
                base_exclude[ciso, o1] = True
                base_exclude[o1, ciso] = True
            if A > ATOM_O2 and cfg.exclude_1_2:
                o2 = idx(i, ATOM_O2)
                base_exclude[ciso, o2] = True
                base_exclude[o2, ciso] = True

    for i in range(L - 1):
        c = idx(i, ATOM_C)
        n_next = idx(i + 1, ATOM_N)
        if cfg.exclude_1_2:
            base_exclude[c, n_next] = True
            base_exclude[n_next, c] = True
        if cfg.exclude_1_3:
            ca = idx(i, ATOM_CA)
            ca_next = idx(i + 1, ATOM_CA)
            base_exclude[ca, n_next] = True
            base_exclude[n_next, ca] = True
            base_exclude[c, ca_next] = True
            base_exclude[ca_next, c] = True

    x_flat = x1_pred.reshape(B * Ns, L * A, 3)
    d = torch.cdist(x_flat, x_flat)
    d = d.reshape(B, Ns, L * A, L * A)
    mask = atom_mask & token_mask[:, :, None]
    mask_flat = mask.reshape(B, L * A)
    ok = mask_flat[:, None, :, None] * mask_flat[:, None, None, :]
    ok = ok & (~torch.eye(L * A, device=dev, dtype=torch.bool)[None, None, :, :])

    iso_idx = iso_acceptor_index.clamp(0, L - 1)
    total = torch.zeros((), device=dev, dtype=x1_pred.dtype)
    count = torch.zeros((), device=dev, dtype=x1_pred.dtype)
    for b in range(B):
        exclude = base_exclude.clone()
        if int(iso_acceptor_index[b].item()) >= 0:
            ciso = idx(int(iso_idx[b].item()), ATOM_CISO)
            n0 = idx(0, ATOM_N)
            if cfg.exclude_1_2:
                exclude[n0, ciso] = True
                exclude[ciso, n0] = True
            if cfg.exclude_1_3:
                if A > ATOM_O1:
                    o1 = idx(int(iso_idx[b].item()), ATOM_O1)
                    exclude[n0, o1] = True
                    exclude[o1, n0] = True
                if A > ATOM_O2:
                    o2 = idx(int(iso_idx[b].item()), ATOM_O2)
                    exclude[n0, o2] = True
                    exclude[o2, n0] = True
        ok_b = ok[b] & (~exclude[None, :, :])
        if cfg.use_softplus:
            pen = F.softplus((r_ij[None, :, :] - d[b]) * float(cfg.beta)).pow(2) / (float(cfg.beta) ** 2)
        else:
            pen = F.relu(r_ij[None, :, :] - d[b]).pow(2)
        pen = pen * ok_b.float() * weight[None, :, :]
        total = total + pen.sum()
        count = count + ok_b.float().sum()
    return total / (count + eps)
