import copy

import numpy as np
import torch
from torch import nn
from typing import List

from ml.aggregators.aggregator import Aggregator
from ml.aggregators.foolsgold import FoolsGold
from ml.aggregators.util import weighted_average


class SybilWallWeightedMedian(Aggregator):

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

            # Add our own model, which we trust
            if own_model is not None:
                weights = np.append(weights, 1)
                models = models + [own_model]

            # Ensure that sum of weights is 1
            weights /= np.sum(weights)

            # Apply the weight vector on this delta
            return WeightedMedian().weighted_median(models, weights)


class WeightedMedian:
    def weighted_median(self, models: List[nn.Module], weights):
        device = torch.device("cpu")
        with torch.no_grad():
            result = copy.deepcopy(models[0])
            result.to(device)
            for p in result.parameters():
                p.mul_(0)
            for model in models:
                model.to(device)

            parameters = np.array(list(map(lambda x: list(x.parameters()), models)), dtype=object)
            for i in range(len(parameters[0])):
                self.recursive_weighted_medians(parameters[:, i], (), np.array(weights), list(result.parameters())[i])
        return result

    def recursive_weighted_medians(self, parameters, index_tuple, weights, result_params):
        p = parameters[0][index_tuple]
        for i in range(len(p)):
            if len(p[i].shape) == 0:
                # Scalar. We can now compute the weighted median
                result_params[index_tuple][i].add_(
                    self.compute_weighted_median(parameters, index_tuple + (i,), weights))
            else:
                # Not a scalar. Recurse
                self.recursive_weighted_medians(parameters, index_tuple + (i,), weights, result_params)

    def compute_weighted_median(self, parameters, index_tuple, weights):
        values = list(map(lambda x: x[index_tuple].item(), parameters))
        return self.weighted_median_2(values, weights)

    def weighted_median_2(self, values, weights):
        i = np.argsort(values)
        c = np.cumsum(weights[i])
        return values[i[np.searchsorted(c, 0.5)]]
