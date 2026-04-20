"""AITR — Cross-Scale Semantic Alignment for image-text retrieval.

Package layout::

    aitr.model        - top-level AITR network (this is what you import)
    aitr.encoders     - image / text encoders
    aitr.prototypes   - semantic prototype banks
    aitr.dim_filter   - IDF + IDE
    aitr.cross_scale  - CSA
    aitr.weak_match   - WMF
    aitr.similarity   - fragment + instance similarity heads
    aitr.loss         - hardest-negative triplet ranking loss
    aitr.utils        - small numerical helpers (l2norm, masked softmax, ...)

Usage::

    from aitr import AITR, AITRConfig, TripletRankingLoss
    model = AITR(AITRConfig(text_encoder="bert"))
"""
from .model import AITR, AITRConfig
from .loss import TripletRankingLoss

__all__ = ["AITR", "AITRConfig", "TripletRankingLoss"]
__version__ = "0.2.0"
