r"""AITR -- top-level network for *Adaptive Image-Text Retrieval*.

Wires together the four AITR ingredients in a single ``nn.Module``::

    encoders -> prototype banks -> IDF -> IDE
              \                              \
               \-> CrossScaleAggregator ----> WMF -> similarity heads

Training-time scoring (factorised, see paper Eq. (train))::

    L_tri  = TripletRanking( lambda1 * S_ini[B,B] + lambda2 * S_ins[B,B] )
    L_aux  = - lambda3 * S_fra[diag].mean()

Evaluation-time scoring (full, see paper Eq. (eval))::

    S_eval[N,M] = lambda1 * S_ini + lambda2 * S_ins + lambda3 * S_fra
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from .encoders import ImageEncoder, BiGRUTextEncoder, BertTextEncoder
from .prototypes import PrototypeBank
from .dim_filter import IntraDimFilter, InterDimExpander
from .cross_scale import CrossScaleAggregator
from .similarity import FragmentSimilarity, InstanceSimilarity
from .weak_match import WeakMatchFilter
from .utils import l2norm


@dataclass
class AITRConfig:
    img_in_dim: int = 2048
    embed_dim: int = 1024
    text_encoder: str = "bigru"          # "bigru" or "bert"
    vocab_size: int = 10_000
    bert_name: str = "bert-base-uncased"
    n_proto: int = 100
    tau: int = 100
    csa_windows: tuple = (1, 2, 3, 4, 5)
    csa_strides: tuple = (1, 1, 2, 2, 3)
    csa_alpha: float = 0.4
    csa_top_pairs: int = 10
    z_balanced: float = 0.4
    z_unbalanced: float = 0.2
    lambda_softmax: float = 9.0
    lambdas: tuple = (0.5, 0.1, 0.1)     # (S_ini, S_ins, S_fra)
    eval_chunk: int = 128                # chunk size for chunked eval


class AITR(nn.Module):
    """The full AITR network."""

    def __init__(self, cfg: AITRConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.img_enc = ImageEncoder(cfg.img_in_dim, cfg.embed_dim)
        if cfg.text_encoder == "bigru":
            self.txt_enc = BiGRUTextEncoder(cfg.vocab_size,
                                            embed_dim=300,
                                            hidden_dim=cfg.embed_dim)
        elif cfg.text_encoder == "bert":
            self.txt_enc = BertTextEncoder(cfg.bert_name, cfg.embed_dim)
        else:
            raise ValueError(cfg.text_encoder)

        self.proto_v = PrototypeBank(cfg.n_proto, cfg.embed_dim)
        self.proto_t = PrototypeBank(cfg.n_proto, cfg.embed_dim)

        self.idf_v = IntraDimFilter(cfg.n_proto, cfg.embed_dim, tau=cfg.tau)
        self.idf_t = IntraDimFilter(cfg.n_proto, cfg.embed_dim, tau=cfg.tau)
        self.ide   = InterDimExpander()

        self.csa = CrossScaleAggregator(cfg.embed_dim,
                                        windows=cfg.csa_windows,
                                        strides=cfg.csa_strides,
                                        alpha=cfg.csa_alpha,
                                        top_pairs=cfg.csa_top_pairs)
        self.wmf = WeakMatchFilter(cfg.z_balanced, cfg.z_unbalanced)
        self.frag_sim = FragmentSimilarity(cfg.lambda_softmax, self.wmf)
        self.inst_sim = InstanceSimilarity()

    # ---------------------------------------------------------------- encode
    def encode_image(self, region_feats: torch.Tensor) -> torch.Tensor:
        return self.img_enc(region_feats)

    def encode_text(self, *args, **kwargs) -> torch.Tensor:
        return self.txt_enc(*args, **kwargs)

    # --------------------------------------------------------------- masks
    def _ide_mask(self) -> torch.Tensor:
        mask_v = self.idf_v(self.proto_v.running_mean)
        mask_t = self.idf_t(self.proto_t.running_mean)
        return self.ide(mask_v, mask_t).clamp(0, 1)         # (d,)

    # ------------------------------------------------------------- forward
    def forward(self,
                region_feats: torch.Tensor,
                text_args: dict) -> dict:
        """Training-time forward.

        Returns a dict with::

            s_ini_mat : (B, B) instance-instance similarity, no IDE mask
            s_ins_mat : (B, B) IDE-masked instance similarity
            s_fra_diag: (B,)   per-pair fragment-level score (matched pairs)
            ide_mask  : (d,)   the IDE mask used in this batch
        """
        v_tokens = self.encode_image(region_feats)            # (B, R, d)
        t_tokens = self.encode_text(**text_args)              # (B, U, d)

        v_inst = l2norm(v_tokens.mean(dim=1), dim=-1)
        t_inst = l2norm(t_tokens.mean(dim=1), dim=-1)

        q_v = self.proto_v.assign(v_inst)
        q_t = self.proto_t.assign(t_inst)
        if self.training:
            self.proto_v.update_running(v_inst, q_v)
            self.proto_t.update_running(t_inst, q_t)

        ide_mask = self._ide_mask()                           # (d,)

        v_inst_m = v_inst * ide_mask
        t_inst_m = t_inst * ide_mask

        # CSA on text tokens (cheap and useful even at training time)
        t_units = self.csa(t_tokens) if t_tokens.size(1) >= 2 else t_tokens

        v_tokens_m = v_tokens * ide_mask                       # (B, R, d)
        t_units_m  = t_units * ide_mask                        # (B, U, d)

        s_ini_mat = v_inst @ t_inst.T                          # (B, B)
        s_ins_mat = l2norm(v_inst_m, -1) @ l2norm(t_inst_m, -1).T

        # Per paper Alg. 4: IDE-masked tokens compute A_{ru} (attention
        # weights); unmasked tokens compute h_r and cos(V_r, h_r).
        s_fra_diag = self.frag_sim(v_tokens_m, t_units_m,
                                   v_tokens, t_units)          # (B,)

        return {"s_ini_mat": s_ini_mat,
                "s_ins_mat": s_ins_mat,
                "s_fra_diag": s_fra_diag,
                "ide_mask": ide_mask}

    # ------------------------------------------------------------ pairwise
    @torch.no_grad()
    def pairwise_similarity(self,
                            v_tokens_all: torch.Tensor,
                            t_tokens_all: torch.Tensor,
                            chunk: int | None = None) -> torch.Tensor:
        """Compute the full (N_img, N_txt) similarity matrix.

        Uses the same convex combination as training (Eq. final), but
        with the fragment branch computed honestly via chunked
        cross-attention.  This is the function called by ``eval.py``.

        Parameters
        ----------
        v_tokens_all : (N_img, R, d)
            All image region tokens (already encoded and on device).
        t_tokens_all : (N_txt, U, d)
            All text tokens (already encoded and on device).
        chunk : optional int
            Chunk size along the image axis (defaults to cfg.eval_chunk).
        """
        l1, l2, l3 = self.cfg.lambdas
        chunk = chunk or self.cfg.eval_chunk
        N_img = v_tokens_all.size(0)
        N_txt = t_tokens_all.size(0)

        v_inst = l2norm(v_tokens_all.mean(dim=1), dim=-1)
        t_inst = l2norm(t_tokens_all.mean(dim=1), dim=-1)

        ide_mask = self._ide_mask()
        v_inst_m = v_inst * ide_mask
        t_inst_m = t_inst * ide_mask

        s_ini = v_inst @ t_inst.T                                       # (N_img, N_txt)
        s_ins = l2norm(v_inst_m, -1) @ l2norm(t_inst_m, -1).T

        # CSA on text tokens once (deterministic and shared across images)
        if t_tokens_all.size(1) >= 2:
            # CSA can be memory-heavy if N_txt is large; chunk along the text axis
            t_units = self._chunked_csa(t_tokens_all, chunk)
        else:
            t_units = t_tokens_all
        # Apply IDE mask symmetrically to both modalities (paper Alg. 1,
        # line A_{ru} = (V_r \odot U^{yz}) * (\tilde{T}_u \odot U^{yz})).
        v_tokens_m = v_tokens_all * ide_mask                            # (N_img, R, d)
        t_units_m  = t_units * ide_mask                                 # (N_txt, U, d)

        s_fra = v_tokens_all.new_zeros(N_img, N_txt)
        for i0 in range(0, N_img, chunk):
            for j0 in range(0, N_txt, chunk):
                s_fra[i0:i0 + chunk, j0:j0 + chunk] = (
                    self.frag_sim.pairwise(
                        v_tokens_m[i0:i0 + chunk],
                        t_units_m[j0:j0 + chunk],
                        v_tokens_all[i0:i0 + chunk],
                        t_units[j0:j0 + chunk])
                )
        return l1 * s_ini + l2 * s_ins + l3 * s_fra

    # -------------------------------------------------------- internal csa
    def _chunked_csa(self,
                     tokens: torch.Tensor,
                     chunk: int) -> torch.Tensor:
        """Apply CSA in slices of size ``chunk`` along the batch axis."""
        out = []
        for i0 in range(0, tokens.size(0), chunk):
            out.append(self.csa(tokens[i0:i0 + chunk]))
        return torch.cat(out, dim=0)
