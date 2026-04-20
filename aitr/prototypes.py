"""Visual / textual semantic prototype banks.

Each modality maintains ``N_p`` learnable prototype vectors and an
EMA-updated relevance distribution that softly assigns samples to
prototypes.  The bank exposes ``assign(samples)`` which returns a
``(batch, N_p)`` posterior used by IDF/IDE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import l2norm


class PrototypeBank(nn.Module):
    def __init__(self, n_proto: int, embed_dim: int,
                 temperature: float = 0.1, ema: float = 0.99) -> None:
        super().__init__()
        self.n_proto = n_proto
        self.tau = temperature
        self.ema = ema
        self.proto = nn.Parameter(torch.randn(n_proto, embed_dim) * 0.02)
        # mean prototype activation, used by IDF for stable channel ranking
        self.register_buffer("running_mean", torch.zeros(n_proto, embed_dim))

    @torch.no_grad()
    def update_running(self, sample_pool: torch.Tensor,
                       posterior: torch.Tensor) -> None:
        # posterior: (B, n_proto), sample_pool: (B, embed_dim)
        weighted = posterior.T @ sample_pool                 # (n_proto, dim)
        denom = posterior.sum(dim=0).clamp_min(1e-6)[:, None]
        new_mean = weighted / denom
        self.running_mean.mul_(self.ema).add_(new_mean, alpha=1 - self.ema)

    def assign(self, samples: torch.Tensor) -> torch.Tensor:
        """Return soft posterior of each sample over prototypes."""
        sims = l2norm(samples) @ l2norm(self.proto).T / self.tau   # (B, n_proto)
        return F.softmax(sims, dim=-1)

    def get_protos(self) -> torch.Tensor:
        return self.proto
