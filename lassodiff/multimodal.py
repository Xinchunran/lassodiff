from __future__ import annotations

import torch


def softmin_aggregate(losses: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    return -tau * torch.logsumexp(-losses / tau, dim=0).mean()
