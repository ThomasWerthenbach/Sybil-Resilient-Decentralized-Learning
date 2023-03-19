from typing import List

import numpy as np
import sklearn.metrics.pairwise as smp
import torch
from torch import nn

from ml.aggregators.aggregator import Aggregator
from ml.aggregators.util import weighted_average


class Repple(Aggregator):
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

    def aggregate(self, models: List[nn.Module], history: List[nn.Module],
                  relevant_parameter_indices: List[int] = None):
        with torch.no_grad():
            parameters = map(lambda x: list(x.parameters()), history)
            parameters = list(map(lambda x: list(map(lambda y: y.data.tolist(), x)), parameters))

            for i in range(len(parameters)):
                while len(parameters[i]) > 0 and any(map(lambda x: type(x) == list, parameters[i])):
                    parameters[i] = self.flatten_one_layer(parameters[i])

            if relevant_parameter_indices:
                relevant_parameters = list(map(lambda x: x[relevant_parameter_indices], parameters))
            else:
                relevant_parameters = parameters

            cs = smp.cosine_similarity(relevant_parameters) - np.eye(
                len(history))  # todo cosine_similarity supports adding an Y param to reduce computations

            max_cs = np.max(cs, axis=1) + 1e-5
            for i in range(len(history)):  # todo We can change one of these range(len()) to models instead of history
                for j in range(len(history)):
                    if i == j:
                        continue
                    if max_cs[i] < max_cs[j]:
                        cs[i][j] = cs[i][j] * max_cs[i] / max_cs[j]

            wv = 1 - (np.max(cs, axis=1))
            wv[wv > 1] = 1
            wv[wv < 0] = 0

            # Rescale so that max value is wv
            wv = wv / np.max(wv)
            wv[(wv == 1)] = .99

            # Logit function
            wv = (np.log((wv / (1 - wv)) + 1e-5) + 0.5)
            wv[(np.isinf(wv) + wv > 1)] = 1
            wv[(wv < 0)] = 0

            # todo krum
            # Ensure that sum of weights is 1
            wv = wv / np.sum(wv)

            # Apply the weight vector on this delta
            return weighted_average(models, wv)
