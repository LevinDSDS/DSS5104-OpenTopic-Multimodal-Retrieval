"""Hardest-negative triplet ranking loss for image-text retrieval."""
from __future__ import annotations

import torch
import torch.nn as nn


class TripletRankingLoss(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """``sim_matrix``: (B, B) similarity scores; diagonal is positive."""
        diagonal = sim_matrix.diag().view(-1, 1)
        d_i2t = diagonal.expand_as(sim_matrix)
        d_t2i = diagonal.t().expand_as(sim_matrix)

        cost_i2t = (self.margin + sim_matrix - d_i2t).clamp(min=0)
        cost_t2i = (self.margin + sim_matrix - d_t2i).clamp(min=0)

        # mask out the diagonal
        eye = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
        cost_i2t = cost_i2t.masked_fill(eye, 0)
        cost_t2i = cost_t2i.masked_fill(eye, 0)

        # hardest negatives along each row / column
        hard_i2t = cost_i2t.max(dim=1)[0]
        hard_t2i = cost_t2i.max(dim=0)[0]
        return (hard_i2t + hard_t2i).mean()
