from typing import List

from torch import nn

from ml.aggregators.krum import Krum
from ml.aggregators.util import weighted_average


class MultiKrum(Krum):
    """
    P. Blanchard, E. M. El Mhamdi, R. Guerraoui, and J. Stainer, “Machine learning with adversaries: Byzantine tolerant gradient descent,”
    in Advances in Neural Information Processing Systems, I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus,
    S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017. [Online]. Available:
    https://proceedings.neurips.cc/paper_files/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf
    """
    def __init__(self, f: int = 1, m: int = 3):
        super().__init__(f)
        self.m = m

    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module],
                  history: List[nn.Module]) -> nn.Module:
        if own_model is not None:
            models = [own_model] + models

        if len(models) == 1:
            # If we have no neighbours, there's nothing to do.
            return models[0]
        if len(models) == 2:
            # If there are only two models, we should use that of the other peer to increase diversity in Non-IID environments
            return models[1]

        # Flatten all parameters
        parameters = map(lambda x: list(x.parameters()), models)
        parameters = list(map(lambda x: list(map(lambda y: y.data.tolist(), x)), parameters))

        # Calculate all pairwise euclidian distances
        distances = self.calculate_distances(parameters)

        # Calculate the score for each model
        scores = []
        for i in range(len(parameters)):
            scores.append(self.krum_score(i, distances))

        # Select the m models with the lowest score
        min_score_indices = []
        for _ in range(min(self.m, len(scores))):
            min_score = min(scores)
            min_score_index = scores.index(min_score)
            min_score_indices.append(min_score_index)
            scores[min_score_index] = float('inf')

        # Calculate the average of the m best models
        weights = [float(1. / len(min_score_indices)) for _ in range(len(min_score_indices))]
        return weighted_average([models[i] for i in min_score_indices], weights)
