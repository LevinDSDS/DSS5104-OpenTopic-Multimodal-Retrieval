import torch

from aitr.cross_scale import CrossScaleAggregator


def test_csa_forward_shape():
    csa = CrossScaleAggregator(embed_dim=32, top_pairs=4)
    tokens = torch.randn(2, 12, 32)
    out = csa(tokens)
    assert out.dim() == 3
    assert out.shape[0] == 2 and out.shape[-1] == 32
    assert out.shape[1] >= 1


def test_csa_handles_short_sequences():
    csa = CrossScaleAggregator(embed_dim=16, top_pairs=2)
    tokens = torch.randn(1, 3, 16)
    out = csa(tokens)
    assert out.shape[0] == 1 and out.shape[-1] == 16
