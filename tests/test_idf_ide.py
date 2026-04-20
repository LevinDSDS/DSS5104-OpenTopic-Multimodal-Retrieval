import torch

from aitr.dim_filter import IntraDimFilter, InterDimExpander


def test_idf_returns_binary_mask_with_correct_sparsity():
    idf = IntraDimFilter(n_proto=4, embed_dim=128, tau=10)
    activation = torch.randn(4, 128)
    mask = idf(activation)
    assert mask.shape == (4, 128)
    assert set(mask.unique().tolist()).issubset({0.0, 1.0})
    assert (mask.sum(dim=-1) == 10).all()


def test_ide_union_aggregated_mask_in_unit_interval():
    mask_v = torch.zeros(3, 8); mask_v[:, :3] = 1
    mask_t = torch.zeros(3, 8); mask_t[:, 5:] = 1
    out = InterDimExpander()(mask_v, mask_t)
    assert out.shape == (8,)
    assert (out >= 0).all() and (out <= 1).all()
