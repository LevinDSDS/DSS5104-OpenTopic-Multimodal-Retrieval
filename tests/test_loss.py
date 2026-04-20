import torch

from aitr.loss import TripletRankingLoss


def test_triplet_loss_zero_when_diagonal_is_max():
    sim = torch.eye(4) + 0.1
    loss = TripletRankingLoss(margin=0.2)(sim)
    assert loss.item() <= 0.21


def test_triplet_loss_positive_when_negatives_dominate():
    sim = torch.zeros(3, 3)
    sim.fill_diagonal_(0.0)
    sim[0, 1] = 0.9
    loss = TripletRankingLoss(margin=0.2)(sim)
    assert loss.item() > 0
