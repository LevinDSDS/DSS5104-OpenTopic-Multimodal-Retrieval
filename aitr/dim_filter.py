"""Intra-modality Dimension Filter (IDF) and Inter-modality
Dimension Expander (IDE).

Given an input feature ``f`` and a prototype-relevance posterior
``q``, IDF first computes the per-prototype channel score by an
EMA-weighted element-wise product, then keeps the top-``tau``
channels per prototype.  IDE then unions the per-prototype masks
of the two modalities to obtain a shared sparse channel selector.

Both modules are written as ``nn.Module`` so that the channel mask
is differentiable with respect to the EMA-tracked prototype
activations (we use a straight-through estimator for the top-k
operation).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .utils import topk_indices


class IntraDimFilter(nn.Module):
    """Channel filter that picks each prototype's top-``tau`` dims."""

    def __init__(self, n_proto: int, embed_dim: int, tau: int = 100) -> None:
        super().__init__()
        self.n_proto = n_proto
        self.embed_dim = embed_dim
        self.tau = tau

    def forward(self, proto_activation: torch.Tensor) -> torch.Tensor:
        """Return a binary mask ``M`` of shape ``(n_proto, embed_dim)``.

        ``proto_activation``: a ``(n_proto, embed_dim)`` tensor that
        averages each prototype's element-wise activations on its
        assigned samples.  See :class:`PrototypeBank.update_running`.
        """
        idx = topk_indices(proto_activation.abs(), k=self.tau, dim=-1)
        mask = torch.zeros_like(proto_activation)
        mask.scatter_(dim=-1, index=idx, value=1.0)
        return mask  # (n_proto, embed_dim)


class InterDimExpander(nn.Module):
    """Union the per-prototype masks of the two modalities."""

    def forward(self,
                mask_v: torch.Tensor,
                mask_t: torch.Tensor,
                pair_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Return the shared mask ``U``.

        ``mask_v``: ``(n_proto_v, d)``, ``mask_t``: ``(n_proto_t, d)``.

        If ``pair_indices`` is None, we average all pairs (instance-
        level many-to-many).  Otherwise it specifies a list of
        ``(i, j)`` pairs for the fragment-level one-to-one branch.
        """
        if pair_indices is None:
            # broadcast union over the cartesian product
            u = mask_v.unsqueeze(1) + mask_t.unsqueeze(0)
            u = (u > 0).float()
            return u.mean(dim=(0, 1))                 # (d,)
        i, j = pair_indices[:, 0], pair_indices[:, 1]
        u = (mask_v[i] + mask_t[j] > 0).float()       # (P, d)
        return u
