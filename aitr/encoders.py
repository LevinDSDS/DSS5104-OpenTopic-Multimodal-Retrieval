"""Image and text encoders for AITR.

The image encoder operates on **pre-computed Faster R-CNN region
features** (36 × 2048) and projects them into the joint embedding
space.  The text encoder either uses a Bi-GRU (parameter-light)
or a frozen / fine-tuned BERT-base (stronger).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .utils import l2norm


# ----------------------------------------------------------------------- image
class ImageEncoder(nn.Module):
    """Linear + LayerNorm projector for pre-computed RoI features."""

    def __init__(self, in_dim: int = 2048, embed_dim: int = 1024) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, region_feats: torch.Tensor) -> torch.Tensor:
        # region_feats: (B, R, in_dim)
        x = self.norm(self.proj(region_feats))
        return l2norm(x, dim=-1)


# ------------------------------------------------------------------------ text
class BiGRUTextEncoder(nn.Module):
    """Embedding + Bi-GRU + linear projection, masked over padding."""

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 300,
                 hidden_dim: int = 1024,
                 num_layers: int = 1,
                 padding_idx: int = 0) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(embed_dim, hidden_dim // 2,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)

    def forward(self,
                token_ids: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L), lengths: (B,)
        x = self.embed(token_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.norm(out)
        return l2norm(out, dim=-1)


class BertTextEncoder(nn.Module):
    """Wrap HuggingFace BERT.  Produces token-level embeddings."""

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 embed_dim: int = 1024,
                 freeze_layers: int = 0) -> None:
        super().__init__()
        try:
            from transformers import BertModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required when text_encoder='bert'."
            ) from exc

        self.bert = BertModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        if freeze_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad_(False)

    def forward(self,
                token_ids: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.bert(input_ids=token_ids,
                        attention_mask=attn_mask).last_hidden_state
        out = self.norm(self.proj(out))
        return l2norm(out, dim=-1)
