import numpy as np
import torch
from torch import nn
from typing import List

from ml.aggregators.aggregator import Aggregator
from ml.aggregators.foolsgold import FoolsGold
from ml.aggregators.median import Median
from ml.aggregators.util import weighted_average


class SybilWallMedian(Aggregator):

    def requires_gossip(self) -> bool:
        return True

    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module], history: List[nn.Module]):
        if len(models) == 0:
            # If we have no neighbours, there's nothing to do.
            return own_model
        if len(models) == 1:
            return weighted_average([own_model, models[0]], [0.5, 0.5])

        with torch.no_grad():
            # Compute the FoolsGold similarity score
            weights = FoolsGold.compute_similarity_score(history)

            # Obtain the indices with the 50% highest weights.
            indices = np.argpartition(weights, -int(len(weights) / 2))[-int(len(weights) / 2):]

            # Map indices to models
            filtered_models = list(map(lambda x: models[x], indices))

            # Compute median
            return Median().aggregate(own_model, None, filtered_models, None)
