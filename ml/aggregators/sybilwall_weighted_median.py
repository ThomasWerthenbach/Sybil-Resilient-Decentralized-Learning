import copy
from typing import List

import numpy as np
import sklearn.metrics.pairwise as smp
import torch
from torch import nn

from ml.aggregators.aggregator import Aggregator
from ml.aggregators.util import weighted_average


class SybilWallWeightedMedian(Aggregator):
    """
    Parts of code adopted from https://github.com/DistributedML/FoolsGold
    """

    def requires_gossip(self) -> bool:
        return True

    def flatten_one_layer(self, l: List) -> List:
        flat_list = []
        for sublist in l:
            if type(sublist) == list:
                for item in sublist:
                    flat_list.append(item)
            else:
                flat_list.append(sublist)
        return flat_list

    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module], history: List[nn.Module],
                  relevant_parameter_indices: List[int] = None):
        if len(models) == 0:
            # If we have no neighbours, there's nothing to do.
            return own_model
        if len(models) == 1:
            return weighted_average([own_model, models[0]], [0.5, 0.5])

        with torch.no_grad():
            parameters = map(lambda x: list(x.parameters()), history)
            parameters = list(map(lambda x: list(map(lambda y: y.data.tolist(), x)), parameters))

            for i in range(len(parameters)):
                while len(parameters[i]) > 0 and any(map(lambda x: type(x) == list, parameters[i])):
                    parameters[i] = self.flatten_one_layer(parameters[i])

            cs = smp.cosine_similarity(parameters) - np.eye(len(history))

            max_cs = np.max(cs, axis=1) + 1e-5
            for i in range(len(history)):
                for j in range(len(history)):
                    if i == j:
                        continue
                    if max_cs[i] < max_cs[j]:
                        cs[i][j] = cs[i][j] * max_cs[i] / max_cs[j]

            wv = 1 - (np.max(cs, axis=1))
            wv = wv[:len(models)]
            wv[wv > 1] = 1
            wv[wv < 0] = 0

            # Rescale so that max value is wv
            wv = wv / np.max(wv)
            wv[(wv == 1)] = .99

            # Logit function
            wv = (np.log((wv / (1 - wv)) + 1e-5) + 0.5)
            wv[(np.isinf(wv) + wv > 1)] = 1
            wv[(wv < 0)] = 0

            # Add our own model, which we trust
            if own_model is not None:
                wv = np.append(wv, 1)
                models = models + [own_model]

            # Ensure that sum of weights is 1
            wv = wv / np.sum(wv)

            # Apply the weight vector on this delta
            return WeightedMedian().weighted_median(models, wv)


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
                result_params[index_tuple][i].add_(self.compute_weighted_median(parameters, index_tuple + (i,), weights))
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
