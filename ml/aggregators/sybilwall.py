import numpy as np
import torch
from torch import nn
from typing import List

from ml.aggregators.aggregator import Aggregator
from ml.aggregators.foolsgold import FoolsGold
from ml.aggregators.util import weighted_average


class SybilWall(Aggregator):
    """
    Implements SybilWall's aggregation function. Exploits similarity between Sybils.
    """

    def requires_gossip(self) -> bool:
        return True

    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module], history: List[nn.Module]):
        """
        Note that the history of gossiped models is already included in the :param history: parameter.
        """

        with torch.no_grad():
            # Compute the FoolsGold similarity score
            weights = FoolsGold.compute_similarity_score(history)

            # Add our own model, which we trust, with a maximum score of 1
            if own_model is not None:
                weights = np.append(weights, 1)
                models = models + [own_model]

            # Weight normalization (sum of weights is 1)
            weights /= np.sum(weights)

            # Weighted average based on the similarity scores.
            return weighted_average(models, weights)
