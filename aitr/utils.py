"""Numerical / tensor utilities used across AITR modules.

Implementation note
-------------------
We deliberately re-implement these primitives from scratch rather
than importing from a reference SCAN/CSA repository, both for
clarity and to make the file independently auditable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def l1norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.abs().sum(dim=dim, keepdim=True) + eps)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def masked_softmax(logits: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1) -> torch.Tensor:
    """Softmax that ignores positions where ``mask == 0``.

    ``mask`` is a 0/1 tensor broadcastable to ``logits``.
    """
    very_neg = torch.finfo(logits.dtype).min / 2
    logits = logits.masked_fill(mask == 0, very_neg)
    return F.softmax(logits, dim=dim)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity.

    ``a``: (N, d), ``b``: (M, d).  Returns (N, M).
    """
    return l2norm(a) @ l2norm(b).T


def topk_indices(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    return x.topk(k=k, dim=dim, largest=True, sorted=False).indices
