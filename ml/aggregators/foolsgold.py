from typing import List

import numpy as np
import sklearn.metrics.pairwise as smp
from torch import nn

from ml.aggregators.aggregator import Aggregator


class FoolsGold(Aggregator):
    """
    todo Foolsgold paper citation
    """

    # todo signature is incorrect, we need the history
    def aggregate(self, models: List[nn.Module], relevant_parameter_indices: List[int] = None):
        parameters = map(lambda x: x.parameters, models)
        flattened_parameters = list(map(lambda x: np.array(x).flatten(), parameters))

        if relevant_parameter_indices:
            relevant_parameters = list(map(lambda x: x[relevant_parameter_indices], flattened_parameters))
        else:
            relevant_parameters = flattened_parameters

        cs = smp.cosine_similarity(relevant_parameters) - np.eye(len(models))

        max_cs = np.max(cs, axis=1) + 1e-5
        for i in range(len(models)):
            for j in range(len(models)):
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

        # Apply the weight vector on this delta
        delta = np.reshape(this_delta, (n, d))

        return np.dot(delta.T, wv)
