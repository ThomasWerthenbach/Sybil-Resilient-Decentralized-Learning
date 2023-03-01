import copy
from typing import List

import torch
from torch import nn


def weighted_average(models: List[nn.Module], weights: List[float]) -> nn.Module:
    """
    Calculates the weighted average
    """
    assert sum(weights) - 1 < 1e-5, "Weights should sum to 1"
    with torch.no_grad():
        avg_delta = copy.deepcopy(models[0])
        for p in avg_delta.parameters():
            p.mul_(0)
        for model, w in zip(models, weights):
            for c1, p1 in zip(avg_delta.parameters(), model.parameters()):
                c1.add_(w * p1)
        return avg_delta
