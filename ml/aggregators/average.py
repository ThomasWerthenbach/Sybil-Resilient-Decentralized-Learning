from typing import List

import torch
from torch import nn

from ml.aggregators.aggregator import Aggregator
from ml.aggregators.util import weighted_average


class Average(Aggregator):
    """
    Basic aggregator which simply aggregates all models
    """

    def aggregate(self, models: List[nn.Module], delta_history: List[nn.Module],
                  relevant_weights: List[int] = None) -> nn.Module:
        with torch.no_grad():
            weights = [float(1. / len(models)) for _ in range(len(models))]
            return weighted_average(models, weights)
