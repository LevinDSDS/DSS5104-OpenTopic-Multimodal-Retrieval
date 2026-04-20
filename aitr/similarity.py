"""Fragment-level and instance-level similarity heads.

``FragmentSimilarity`` exposes both a per-pair scalar score (used as an
auxiliary loss during training, see ``train.py``) and a chunked
pairwise (B1, B2) score matrix (used at evaluation, see
``eval.py`` and ``model.AITR.pairwise_similarity``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import l2norm
from .weak_match import WeakMatchFilter


class FragmentSimilarity(nn.Module):
    """One-to-one cross-modal attention with adaptive weak-match filter.

    Per paper Alg. 4 lines 556-560, the IDE mask is applied **only** when
    computing the attention matrix A_{ru}.  The weighted-sum h_r and the
    final cos(V_r, h_r) operate on the *unmasked* tokens so that the
    full-dimensional semantic signal is preserved in the score.
    """

    def __init__(self, lambda_softmax: float = 9.0,
                 wmf: WeakMatchFilter | None = None) -> None:
        super().__init__()
        self.lam = lambda_softmax
        self.wmf = wmf or WeakMatchFilter()

    # ------------------------------------------------------------ per-pair
    def forward(self,
                regions_m: torch.Tensor,
                words_m:   torch.Tensor,
                regions_raw: torch.Tensor,
                words_raw:   torch.Tensor) -> torch.Tensor:
        """Diagonal (matched) score per batch element.

        ``*_m``  : IDE-masked tokens  (used for attention weights).
        ``*_raw``: unmasked tokens    (used for h_r and cos).
        All tensors have shape (B, seq_len, d).
        """
        B = regions_m.size(0)
        scores = regions_m.new_zeros(B)
        for b in range(B):
            scores[b] = self._single_pair(regions_m[b], words_m[b],
                                          regions_raw[b], words_raw[b])
        return scores

    # ------------------------------------------------------------- pairwise
    def pairwise(self,
                 regions_m: torch.Tensor,
                 words_m:   torch.Tensor,
                 regions_raw: torch.Tensor,
                 words_raw:   torch.Tensor) -> torch.Tensor:
        """Compute the full (B1, B2) similarity matrix.

        Memory cost is ``O(B1*B2*R*U)``; callers should chunk along
        ``B1`` (and/or ``B2``) when the dataset is large -- see
        ``AITR.pairwise_similarity`` for the chunking driver.
        """
        B1, R, d = regions_m.shape
        B2, U, _ = words_m.shape
        sim = torch.einsum('ird,jud->ijru', regions_m, words_m)      # (B1,B2,R,U)

        sim_flat = sim.reshape(B1 * B2, R, U)
        for k in range(B1 * B2):
            sim_flat[k] = self.wmf(sim_flat[k], balanced=True)
        sim = sim_flat.view(B1, B2, R, U)

        attn_w = F.softmax(sim * self.lam, dim=-1)                   # over U
        attn = torch.einsum('ijru,jud->ijrd', attn_w, words_raw)     # unmasked text
        attn = l2norm(attn, dim=-1)
        v_normed = l2norm(regions_raw.unsqueeze(1), dim=-1)           # unmasked image
        score = (v_normed * attn).sum(-1).mean(-1)                    # (B1,B2)
        return score

    # ----------------------------------------------------------- internal
    def _single_pair(self,
                     regions_m_b: torch.Tensor,
                     words_m_b: torch.Tensor,
                     regions_raw_b: torch.Tensor,
                     words_raw_b: torch.Tensor) -> torch.Tensor:
        sim = regions_m_b @ words_m_b.T                               # (R, U)
        sim = self.wmf(sim, balanced=True)
        attn_w = F.softmax(sim * self.lam, dim=1)
        attn = attn_w @ words_raw_b                                   # (R, d)
        attn = l2norm(attn, dim=-1)
        v_normed = l2norm(regions_raw_b, dim=-1)
        return (v_normed * attn).sum(dim=-1).mean()


class InstanceSimilarity(nn.Module):
    """Many-to-many similarity over IDE-masked features.

    Returns either a per-pair (B,) or pairwise (B1, B2) score depending
    on input shapes.
    """

    def forward(self,
                ins_v: torch.Tensor,
                ins_t: torch.Tensor) -> torch.Tensor:
        if ins_v.dim() == 2 and ins_t.dim() == 2 and ins_v.size(0) == ins_t.size(0):
            return (l2norm(ins_v) * l2norm(ins_t)).sum(dim=-1)
        return l2norm(ins_v) @ l2norm(ins_t).T

    def pairwise(self,
                 ins_v: torch.Tensor,
                 ins_t: torch.Tensor) -> torch.Tensor:
        return l2norm(ins_v) @ l2norm(ins_t).T
