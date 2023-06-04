from typing import List

import numpy as np
import sklearn.metrics.pairwise as smp
import torch
from torch import nn

from ml.aggregators.aggregator import Aggregator
from ml.aggregators.util import weighted_average


class FoolsGold(Aggregator):
    """
    C. Fung, C. J. M. Yoon, and I. Beschastnikh, “Mitigating sybils in federated learning poisoning,” CoRR,
    vol. abs/1808.04866, 2018. [Online]. Available: https://arxiv.org/abs/1808.04866

    Parts of code adopted from https://github.com/DistributedML/FoolsGold
    """

    def requires_gossip(self) -> bool:
        return False


    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module], history: List[nn.Module],
                  relevant_parameter_indices: List[int] = None):
        with torch.no_grad():
            if own_model is not None:
                models = models + [own_model]
                history = history + [own_history]
            parameters = map(lambda x: list(x.parameters()), history)
            parameters = list(map(lambda x: list(map(lambda y: y.data.tolist(), x)), parameters))

            self.flatten_all(parameters)

            if relevant_parameter_indices:
                relevant_parameters = list(map(lambda x: x[relevant_parameter_indices], parameters))
            else:
                relevant_parameters = parameters

            cs = smp.cosine_similarity(relevant_parameters) - np.eye(len(history))

            max_cs = np.max(cs, axis=1) + 1e-5
            for i in range(len(history)):
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

            # Ensure that sum of weights is 1
            wv = wv / np.sum(wv)

            # Apply the weight vector on this delta
            return weighted_average(models, wv)
