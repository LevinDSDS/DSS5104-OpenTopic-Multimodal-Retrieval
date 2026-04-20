"""Adaptive Weak-Match Filter (WMF).

Estimate the empirical mean ``mu`` and standard deviation
``sigma`` of the *positive* entries of a cross-modal similarity
matrix.  Threshold by

    b = mu + sigma * sqrt(-2 * ln(z + eps))

where ``z`` is a tail-probability hyper-parameter.
Entries below ``b`` are zeroed out before attention re-weighting.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class WeakMatchFilter(nn.Module):
    def __init__(self, z_balanced: float = 0.4,
                 z_unbalanced: float = 0.2,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.z_balanced = z_balanced
        self.z_unbalanced = z_unbalanced
        self.eps = eps

    def threshold(self, sim: torch.Tensor, balanced: bool) -> torch.Tensor:
        positive = sim[sim > 0]
        if positive.numel() < 4:
            return sim.new_zeros(())
        mu = positive.mean()
        sigma = positive.std(unbiased=False)
        z = self.z_balanced if balanced else self.z_unbalanced
        return mu + sigma * math.sqrt(-2.0 * math.log(z + self.eps))

    def forward(self,
                sim: torch.Tensor,
                balanced: bool = True) -> torch.Tensor:
        b = self.threshold(sim, balanced=balanced)
        return torch.where(sim > b, sim, sim.new_zeros(()))
