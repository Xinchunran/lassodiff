from __future__ import annotations

from dataclasses import dataclass, field
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
    w_iso: float = 10.0
    w_plug: float = 1.0
    w_thr: float = 0.5
    w_clash: float = 0.1
    w_link: float = 0.0
    w_tube: float = 0.0


@dataclass
class IsoCfg:
    dmin: float = 2.0
    dmax: float = 3.0
    angle_min: float = 90.0
    angle_max: float = 140.0
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


def loss_flow(v_pred, v_star, token_mask, w_res=None, atom_mask=None):
    if v_pred.dim() == 5:
        mask = token_mask[:, None, :, None, None].float()
    else:
        mask = token_mask[:, None, :, None].float()
    if atom_mask is not None:
        mask = mask * atom_mask[:, None, :, :, None].float()
    diff2 = (v_pred - v_star) ** 2
    if w_res is not None:
        if w_res.dim() == 1:
            w_ = w_res[None, None, :, None]
        else:
            w_ = w_res[:, None, :, None]
        if diff2.dim() == 5:
            w_ = w_[:, :, :, None, None]
        diff2 = diff2 * w_
    return (diff2 * mask).sum() / (mask.sum() * 3.0 + 1e-8)


def loss_bb(x1_pred, x1_true, token_mask, w_res=None, atom_mask=None):
    if x1_pred.dim() == 5:
        mask = token_mask[:, None, :, None, None].float()
    else:
        mask = token_mask[:, None, :, None].float()
    if atom_mask is not None:
        mask = mask * atom_mask[:, None, :, :, None].float()
    diff2 = (x1_pred - x1_true) ** 2
    if w_res is not None:
        if w_res.dim() == 1:
            w_ = w_res[None, None, :, None]
        else:
            w_ = w_res[:, None, :, None]
        if diff2.dim() == 5:
            w_ = w_[:, :, :, None, None]
        diff2 = diff2 * w_
    return (diff2 * mask).sum() / (mask.sum() * 3.0 + 1e-8)


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
