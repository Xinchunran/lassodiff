from __future__ import annotations

from dataclasses import dataclass, field
import torch

from .flow_matching import SigmaSchedule
from .losses_lasso import (
    IsoCfg,
    PlugCfg,
    LinkCfg,
    TubeCfg,
    ATOM_CA,
    ATOM_N,
    ATOM_CISO,
    gauss_linking_integral_ca,
    loss_link_ca,
    loss_plug_thread_ca,
    loss_ring_tube_ca,
    project_iso_bb,
)


@dataclass
class TopologyConfig:
    apply_projection: bool = True
    iso_cfg: IsoCfg = field(default_factory=IsoCfg)
    plug_cfg: PlugCfg = field(default_factory=PlugCfg)
    link_cfg: LinkCfg = field(default_factory=LinkCfg)
    tube_cfg: TubeCfg = field(default_factory=TubeCfg)
    w_plug: float = 1.0
    w_thr: float = 0.5
    w_link: float = 0.0
    w_tube: float = 0.0
    guidance_eta: float = 0.1
    guard_link_jump: bool = False
    link_jump_delta: float = 0.3
    guard_strategy: str = "adaptive_step"
    max_retry: int = 1


@dataclass
class SamplerConfig:
    steps: int = 40
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    return_metrics: bool = False


def sample_rectified_flow(
    model,
    seqs,
    aa_ids,
    token_mask,
    k,
    p,
    atom_mask=None,
    iso_acceptor_index=None,
    cfg: SamplerConfig = SamplerConfig(),
):
    device = aa_ids.device
    B, L = aa_ids.shape
    Ns = 1
    A = getattr(model, "n_atom", 7)
    x = torch.randn(B, Ns, L, A, 3, device=device) * 5.0
    sched = SigmaSchedule()
    topo_cfg = cfg.topology
    link_prev = None
    topo_metrics = {}

    for i in range(cfg.steps):
        t = torch.full((B, Ns), i / (cfg.steps - 1), device=device)
        sigma = sched.sigma(t)
        with torch.no_grad():
            v = model(
                seqs=seqs,
                aa_ids=aa_ids,
                token_mask=token_mask,
                k=k,
                p=p,
                x_t=x,
                sigma=sigma,
                t_scalar=t,
            )
        if cfg.steps > 1:
            dt = 1.0 / (cfg.steps - 1)
        else:
            dt = 1.0

        x_prev = x
        use_guidance = topo_cfg.guidance_eta > 0.0 and (
            topo_cfg.w_plug > 0.0 or topo_cfg.w_thr > 0.0 or topo_cfg.w_link > 0.0 or topo_cfg.w_tube > 0.0
        )
        if use_guidance:
            x_in = x.detach().requires_grad_(True)
            x_in_ca = x_in[..., ATOM_CA, :]
            L_plug, L_thr = loss_plug_thread_ca(x_in_ca, k, p, token_mask, plug_cfg=topo_cfg.plug_cfg)
            L_link = loss_link_ca(x_in_ca, k, p, token_mask, link_cfg=topo_cfg.link_cfg) if topo_cfg.w_link != 0.0 else x_in.new_tensor(0.0)
            L_tube = loss_ring_tube_ca(x_in_ca, k, p, token_mask, tube_cfg=topo_cfg.tube_cfg) if topo_cfg.w_tube != 0.0 else x_in.new_tensor(0.0)
            guide_loss = topo_cfg.w_plug * L_plug + topo_cfg.w_thr * L_thr + topo_cfg.w_link * L_link + topo_cfg.w_tube * L_tube
            g = torch.autograd.grad(guide_loss, x_in, create_graph=False)[0]
            x = (x_in + dt * (v - topo_cfg.guidance_eta * g)).detach()
        else:
            x = x + dt * v

        if topo_cfg.apply_projection and atom_mask is not None and iso_acceptor_index is not None:
            x = project_iso_bb(x, iso_acceptor_index, token_mask, atom_mask, iso_cfg=topo_cfg.iso_cfg)

        if topo_cfg.guard_link_jump:
            with torch.no_grad():
                link = gauss_linking_integral_ca(x, k, p, token_mask, link_cfg=topo_cfg.link_cfg)
            if link_prev is not None:
                jump = (link - link_prev).abs().max().item()
                if jump > topo_cfg.link_jump_delta:
                    if topo_cfg.guard_strategy == "adaptive_step" and topo_cfg.max_retry > 0:
                        dt_retry = dt * 0.5
                        x = x_prev
                        if use_guidance:
                            x_in = x.detach().requires_grad_(True)
                            x_in_ca = x_in[..., ATOM_CA, :]
                            L_plug, L_thr = loss_plug_thread_ca(x_in_ca, k, p, token_mask, plug_cfg=topo_cfg.plug_cfg)
                            L_link = loss_link_ca(x_in_ca, k, p, token_mask, link_cfg=topo_cfg.link_cfg) if topo_cfg.w_link != 0.0 else x_in.new_tensor(0.0)
                            L_tube = loss_ring_tube_ca(x_in_ca, k, p, token_mask, tube_cfg=topo_cfg.tube_cfg) if topo_cfg.w_tube != 0.0 else x_in.new_tensor(0.0)
                            guide_loss = topo_cfg.w_plug * L_plug + topo_cfg.w_thr * L_thr + topo_cfg.w_link * L_link + topo_cfg.w_tube * L_tube
                            g = torch.autograd.grad(guide_loss, x_in, create_graph=False)[0]
                            x = (x_in + dt_retry * (v - topo_cfg.guidance_eta * g)).detach()
                        else:
                            x = x + dt_retry * v
                        if topo_cfg.apply_projection and atom_mask is not None and iso_acceptor_index is not None:
                            x = project_iso_bb(x, iso_acceptor_index, token_mask, atom_mask, iso_cfg=topo_cfg.iso_cfg)
                        with torch.no_grad():
                            link = gauss_linking_integral_ca(x[..., ATOM_CA, :], k, p, token_mask, link_cfg=topo_cfg.link_cfg)
                    else:
                        x = x_prev
                        link = link_prev
            link_prev = link

        x = x * token_mask[:, None, :, None, None].float()
    x = x * token_mask[:, None, :, None, None].float()
    if not cfg.return_metrics:
        return x[:, 0]

    with torch.no_grad():
        b = torch.arange(B, device=device)
        k_clamped = k.clamp(0, L - 1)
        p_clamped = p.clamp(0, L - 1)
        if atom_mask is not None and iso_acceptor_index is not None:
            idx = iso_acceptor_index.clamp(0, L - 1)
            iso_dist = torch.linalg.vector_norm(x[:, 0, 0, ATOM_N, :] - x[b, 0, idx, ATOM_CISO, :], dim=-1)
        else:
            iso_dist = torch.linalg.vector_norm(x[:, 0, 0, ATOM_CA, :] - x[b, 0, k_clamped, ATOM_CA, :], dim=-1)
        x_ca = x[..., ATOM_CA, :]
        L_plug, L_thr = loss_plug_thread_ca(x_ca, k, p, token_mask, plug_cfg=topo_cfg.plug_cfg)
        link = gauss_linking_integral_ca(x_ca, k, p, token_mask, link_cfg=topo_cfg.link_cfg)
        topo_metrics = {
            "iso_distance": iso_dist,
            "plug_score": L_plug,
            "thread_score": L_thr,
            "link_score": link,
        }
    return x[:, 0], topo_metrics
