import torch


def sinusoidal_pos1d(Lmax: int, dim: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(Lmax, dim, device=device)
    position = torch.arange(0, Lmax, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device)
        * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
