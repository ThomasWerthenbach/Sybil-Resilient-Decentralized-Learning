import copy
from typing import List

import torch
from torch import nn

from ml.aggregators.aggregator import Aggregator
from ml.aggregators.util import weighted_average


class Median(Aggregator):
    """
    http://proceedings.mlr.press/v80/yin18a/yin18a.pdf
    """
    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module],
                  delta_history: List[nn.Module], relevant_weights: List[int] = None) -> nn.Module:
        if own_model is not None:
            models = [own_model] + models
        if len(models) == 1:
            # If we have no neighbours, there's nothing to do.
            return models[0]
        if len(models) == 2:
            return weighted_average([models[0], models[1]], [0.5, 0.5])

        # Get coordinate-wise median from all models
        parameters = list(map(lambda x: list(x.parameters()), models))
        with torch.no_grad():
            avg_delta = copy.deepcopy(models[0])
            for p in avg_delta.parameters():
                p.mul_(0)
            for i in range(len(parameters[0])):
                # Get coordinate-wise median from all models
                median = torch.median(torch.stack([p[i].data for p in parameters]), dim=0)[0]
                list(avg_delta.parameters())[i].add_(median)
            return avg_delta

    def requires_gossip(self) -> bool:
        return False