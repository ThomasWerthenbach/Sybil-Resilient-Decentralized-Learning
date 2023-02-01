import copy
from typing import List

import torch
from torch import nn


def weighted_average(deltas: List[nn.Module], weights: List[float]) -> nn.Module:
    """
    Calculates the weighted average
    """
    with torch.no_grad():
        avg_delta = copy.deepcopy(deltas[0])
        for p in avg_delta.parameters():
            p.mul_(0)
        for delta, w in zip(deltas, weights):
            for c1, p1 in zip(avg_delta.parameters(), delta.parameters()):
                c1.add_(w * p1)
        return avg_delta
