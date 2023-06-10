import numpy as np
import torch
from torch import nn
from typing import List

from ml.aggregators.aggregator import Aggregator
from ml.aggregators.foolsgold import FoolsGold
from ml.aggregators.krum import Krum
from ml.aggregators.util import weighted_average


class SybilWallKrumFilter(Aggregator):

    def requires_gossip(self) -> bool:
        return True

    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module], history: List[nn.Module]):
        if len(models) == 0:
            # If we have no neighbours, there's nothing to do.
            return own_model
        if len(models) == 1:
            # If we have only one neighbour, we can average our own model with theirs.
            return weighted_average([own_model, models[0]], [0.5, 0.5])

        with torch.no_grad():
            # Compute the FoolsGold similarity score
            weights = FoolsGold.compute_similarity_score(history)

            # Filter out the model with the lowest krum score
            if len(models) > 1:
                parameters = map(lambda x: list(x.parameters()), models)
                parameters = list(map(lambda x: list(map(lambda y: y.data.tolist(), x)), parameters))
                krum = Krum(f=1)
                scores = krum.calculate_scores(parameters)
                max_score = max(scores)
                max_score_index = scores.index(max_score)
                weights[max_score_index] = 0

            # Add our own model, which we trust, with a maximum score of 1
            if own_model is not None:
                weights = np.append(weights, 1)
                models = models + [own_model]

            # Weight normalization (sum of weights is 1)
            weights /= np.sum(weights)

            # Weighted average based on the similarity scores.
            return weighted_average(models, weights)
