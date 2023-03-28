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

    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module], history: List[nn.Module],
                  relevant_parameter_indices: List[int] = None):
        """
        We should only do cosine similarity on the parameters from other nodes, because we know that we are good, and
        we should not punish users who look like us.

        1. Perform cosine similarity on the parameters from other nodes
        2. Find the maximum cosine similarity for each node
        3.
        4. Compute the cosine similarity between our model history and the other nodes
        5. Compute the weight vector
        """

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
                np.append(wv, 1)
                models = models + [own_model]

            # Ensure that sum of weights is 1
            wv = wv / np.sum(wv)

            # Apply the weight vector on this delta
            return weighted_average(models, wv)
