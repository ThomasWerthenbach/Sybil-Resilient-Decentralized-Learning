import math
import sys
from typing import Tuple, List

import numpy as np
import torch.nn as nn

from ml.ml_settings import MLSettings


class BristleIntegrator:
    def __init__(self,
                 own_model: nn.Module,
                 models: Tuple[nn.Module],
                 dataset,
                 classes: List[int],
                 n_classes: int,
                 settings: MLSettings):
        self.own_model = own_model
        self.models = models
        self.dataset = dataset
        self.classes = classes
        self.n_classes = n_classes
        self.settings = settings

    def integrate(self) -> nn.Module:
        """
        Integrate the models based on their performance.
        """
        # Compute the performance of each model
        performances: List[List[float]] = []

        models: List[nn.Module] = list(self.models)
        models.append(self.own_model)

        # calculate performances
        for model in models:
            performance: List[float] = [0.0] * self.n_classes
            for c in self.classes:
                performance[c] = self.evaluate_model(model, c)
            performances.append(performance)

        certainty: List[float] = [0.0] * self.n_classes
        disc: List[List[float]] = [[0.0] * self.n_classes for _ in range(len(self.models))]
        faWg: List[List[float]] = [[0.0] * self.n_classes for _ in range(len(self.models))]
        foWg: List[float] = [0.0] * self.n_classes
        # calculate weights by Bristle
        for i, _ in enumerate(self.models):
            # Get phi best performing classes for model
            best = sorted(performances[i])[:self.settings.phi]
            certainty[i] = max(sum(best) / len(best) - np.std(best), 0.0)
            for c in self.classes:
                if performances[i][c] >= performances[len(performances) - 1][c]:
                    disc[i][c] = ((performances[i][c] - performances[len(performances) - 1][c])
                                  * self.settings.theta) ** 3
                else:
                    disc[i][c] = -sys.maxsize
                faWg[i][c] = self.sigmoid_familiar(disc[i][c], certainty[i])
            foWg[i] += self.sigmoid_foreign(sum(disc[i]), certainty[i])

        # todo calculate new model through weighted average.
        return average_model

    def sigmoid_familiar(self, score: float, certainty: float) -> float:
        return max(0.0, self.settings.familiar_w1 / (1 + math.exp(-score / 100.0)) - self.settings.familiar_w2) * certainty

    def sigmoid_foreign(self, score: float, certainty: float) -> float:
        return max(0.0, self.settings.foreign_w1 / (1 + math.exp(-score / 100.0)) - self.settings.foreign_w2) * certainty

    def evaluate_model(self, model, c) -> float:
        pass
