from __future__ import annotations

from dataclasses import dataclass
import torch


def right_pad_dims_to(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.reshape(*t.shape, *((1,) * padding_dims))


class BasePath:
    def compute_alpha_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def compute_sigma_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def compute_d_alpha_alpha_ratio_t(self, t: torch.Tensor) -> torch.Tensor:
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        return d_alpha_t / alpha_t

    def compute_mu_t(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def interpolant(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = right_pad_dims_to(x0, t)
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1)
        return t, xt, ut


class LinearPath(BasePath):
    def compute_alpha_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return t, torch.ones_like(t)

    def compute_sigma_t(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 1.0 - t, -torch.ones_like(t)


@dataclass
class SigmaSchedule:
    sigma_data: float = 16.0
    s_max: float = 160.0
    s_min: float = 4e-4
    p: float = 7.0

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        smax_ = self.s_max ** (1.0 / self.p)
        smin_ = self.s_min ** (1.0 / self.p)
        base = smax_ + t * (smin_ - smax_)
        return self.sigma_data * (base**self.p)


def sample_t(B: int, Ns: int, device: torch.device) -> torch.Tensor:
    return torch.rand(B, Ns, device=device)


def flow_interpolate(x0, x1, t):
    path = LinearPath()
    _, x_t, v_star = path.interpolant(t, x0, x1)
    return x_t, v_star
