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

    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module], history: List[nn.Module]):
        if own_model is not None:
            models = models + [own_model]
            history = history + [own_history]

        with torch.no_grad():
            weights = self.compute_similarity_score(history)

            # Weight normalization (sum of weights is 1)
            weights /= np.sum(weights)

            # Weighted average based on the similarity scores.
            return weighted_average(models, weights)

    @staticmethod
    def compute_similarity_score(history: List[nn.Module]):
        """
        This function represents the main FoolsGold inspired component used within SybilWall.

        Computes the similarity score based on the FoolsGold algorithm.
        """
        parameters = map(lambda x: list(x.parameters()), history)
        parameters = list(map(lambda x: list(map(lambda y: y.data.tolist(), x)), parameters))

        Aggregator.flatten_all(parameters)

        # Compute similarity matrix
        similarity = smp.cosine_similarity(parameters)
        similarity -= np.eye(len(history))

        # Apply pardoning (see FoolsGold paper)
        maximum_similarity = np.max(similarity, axis=1) + 1e-5
        for i in range(len(history)):
            for j in list(range(0, i)) + list(range(i + 1, len(history))):
                if maximum_similarity[i] < maximum_similarity[j]:
                    similarity[i][j] *= maximum_similarity[i] / maximum_similarity[j]

        # Compute new maximum weights
        weights = 1 - np.max(similarity, axis=1)

        # Clip weights
        weights = np.clip(weights, 0, 1)

        # Rescale weights
        weights = weights / np.max(weights)

        # Logit function
        weights = np.log(weights / (1 - weights)) + 0.5

        # Clip weights.
        # Note that np.clip handles +inf and -inf.
        return np.clip(weights, 0, 1)
