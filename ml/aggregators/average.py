from typing import List

import torch
from torch import nn

from ml.aggregators.aggregator import Aggregator
from ml.aggregators.util import weighted_average


class Average(Aggregator):
    """
    Basic aggregator which simply aggregates all models
    """

    def requires_gossip(self) -> bool:
        return False

    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module], history: List[nn.Module]) -> nn.Module:
        if own_model is not None:
            models = models + [own_model]
        weights = [float(1. / len(models)) for _ in range(len(models))]
        return weighted_average(models, weights)
