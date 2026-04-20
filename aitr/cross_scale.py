"""Cross-Scale Aggregator (CSA).

Generates two complementary families of *unit* embeddings on top
of token features:

1. **Position-aware multi-window subsequences** -- sliding windows
   of sizes ``W = (1, 2, 3, 4, 5)`` and strides ``(1, 1, 2, 2, 3)``.
2. **Co-occurrence-aware adaptive subsequences** -- greedily chained
   tokens whose pair-wise cosine similarity exceeds ``alpha``.

The two families are then paired through an Intersection-over-Union
(IoU) on token indices and the top-``P`` pairs are concatenated and
projected back to the joint embedding space.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import l2norm


class CrossScaleAggregator(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 windows: Tuple[int, ...] = (1, 2, 3, 4, 5),
                 strides: Tuple[int, ...] = (1, 1, 2, 2, 3),
                 alpha: float = 0.4,
                 max_chain_len: int = 5,
                 top_pairs: int = 10) -> None:
        super().__init__()
        assert len(windows) == len(strides)
        self.windows = windows
        self.strides = strides
        self.alpha = alpha
        self.max_chain_len = max_chain_len
        self.top_pairs = top_pairs
        self.proj = nn.Linear(embed_dim * 2, embed_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    # --------------------------- 1. position-aware --------------------------
    def position_subseqs(self,
                         tokens: torch.Tensor) -> List[torch.Tensor]:
        """Return a list of sub-tensors per window/stride combo.

        Each entry has shape ``(B, n_subseq_w, w, d)``.
        """
        B, L, d = tokens.shape
        out = []
        for w, s in zip(self.windows, self.strides):
            if L < w:
                continue
            starts = torch.arange(0, L - w + 1, s, device=tokens.device)
            # build (B, n, w, d)
            sub = torch.stack(
                [tokens[:, st:st + w, :] for st in starts.tolist()], dim=1)
            out.append(sub)
        return out

    # ------------------------- 2. co-occurrence ------------------------------
    def cooccur_subseqs(self,
                        tokens: torch.Tensor) -> torch.Tensor:
        """Return ``(B, L, max_chain_len, d)`` greedy chains.

        For token i the chain greedily appends tokens with the
        largest cosine similarity above ``alpha`` until the chain
        is full or no candidate remains.
        """
        B, L, d = tokens.shape
        sims = l2norm(tokens) @ l2norm(tokens).transpose(-1, -2)   # (B, L, L)
        # forbid self-chaining
        sims = sims - torch.eye(L, device=tokens.device).unsqueeze(0) * 9.0

        chains = tokens.new_zeros(B, L, self.max_chain_len, d)
        chains[:, :, 0, :] = tokens
        used = torch.zeros(B, L, L, dtype=torch.bool, device=tokens.device)
        used.scatter_(2, torch.arange(L, device=tokens.device)
                          .view(1, L, 1).expand(B, -1, -1), True)
        for step in range(1, self.max_chain_len):
            sims_masked = sims.masked_fill(used, -9.0)
            best_val, best_idx = sims_masked.max(dim=-1)         # (B, L)
            stop = best_val < self.alpha
            gather_idx = best_idx.unsqueeze(-1).expand(-1, -1, d)
            picked = torch.gather(tokens, dim=1, index=gather_idx)  # (B, L, d)
            picked = torch.where(stop.unsqueeze(-1), tokens, picked)
            chains[:, :, step, :] = picked

            # vectorised update of the (B, L, L) used mask
            update = F.one_hot(best_idx, num_classes=L).bool()      # (B, L, L)
            used = used | update
        return chains

    # ----------------------------- 3. fuse ----------------------------------
    def fuse(self,
             pos_subseqs: List[torch.Tensor],
             cooc_subseqs: torch.Tensor) -> torch.Tensor:
        """Pair pos vs co-occurrence subsequences with IoU and fuse.

        Returns a unit-tensor ``(B, n_units, d)`` ready for similarity.
        """
        B = cooc_subseqs.shape[0]
        d = cooc_subseqs.shape[-1]
        units = []
        for sub in pos_subseqs:
            # pool the position subseq to a vector
            pos_vec = sub.mean(dim=2)                # (B, n_pos, d)
            # pool the co-occurrence chain to a vector
            cooc_vec = cooc_subseqs.mean(dim=2)      # (B, L,    d)
            # pairwise dot-product as IoU surrogate
            iou = l2norm(pos_vec) @ l2norm(cooc_vec).transpose(-1, -2)
            top_iou, top_idx = iou.flatten(1).topk(
                k=min(self.top_pairs, iou.numel() // B), dim=-1)
            n_pos, n_cooc = pos_vec.size(1), cooc_vec.size(1)
            for k in range(top_iou.size(1)):
                pi = (top_idx[:, k] // n_cooc).view(B, 1, 1).expand(-1, 1, d)
                ci = (top_idx[:, k] %  n_cooc).view(B, 1, 1).expand(-1, 1, d)
                p_vec = torch.gather(pos_vec, 1, pi).squeeze(1)
                c_vec = torch.gather(cooc_vec, 1, ci).squeeze(1)
                units.append(self.proj(torch.cat([p_vec, c_vec], dim=-1)))
        if not units:
            # degenerate: just project the original tokens (shouldn't happen)
            return cooc_subseqs.mean(dim=2)
        out = torch.stack(units, dim=1)              # (B, n_units, d)
        return l2norm(out, dim=-1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.position_subseqs(tokens),
                         self.cooccur_subseqs(tokens))
