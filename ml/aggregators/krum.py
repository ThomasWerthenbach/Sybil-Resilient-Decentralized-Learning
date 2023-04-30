from typing import List

from torch import nn

from ml.aggregators.aggregator import Aggregator


class Krum(Aggregator):
    """
    https://proceedings.neurips.cc/paper_files/paper/2017/file/f4b9ec30ad9f68f89b29639786cb62ef-Paper.pdf
    """
    def __init__(self, f: int = 1):
        self.f = f

    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module],
                  delta_history: List[nn.Module], relevant_weights: List[int] = None) -> nn.Module:
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

        # Calculate the score for each model
        scores = self.calculate_scores(parameters)

        # Select the model with the lowest score
        min_score = min(scores)
        min_score_index = scores.index(min_score)
        return models[min_score_index]

    def requires_gossip(self) -> bool:
        return False

    def calculate_scores(self, parameters):
        # Calculate all pairwise euclidian distances
        distances = self.calculate_distances(parameters)

        # Calculate the score for each model
        scores = []
        for i in range(len(parameters)):
            scores.append(self.krum_score(i, distances))
        return scores

    def calculate_distances(self, parameters):
        self.flatten_all(parameters)
        distances = []
        for i in range(len(parameters)):
            distances.append([])
            for j in range(len(parameters)):
                distances[i].append(self.euclidian_distance(parameters[i], parameters[j]))
        return distances

    def euclidian_distance(self, param: List[float], param1: List[float]):
        s = 0
        for i in range(len(param)):
            s += (param[i] - param1[i]) ** 2
        return s ** 0.5

    def krum_score(self, i, distances):
        # Get the distance to all other models
        distances_i = distances[i]
        # Sort the distances
        distances_i.sort()
        # Get the n-f-2 smallest distances
        distances_i = distances_i[:len(distances_i) - self.f - 2]
        # Square the distances
        distances_i = list(map(lambda x: x ** 2, distances_i))
        # Calculate the average of the f+1 smallest distances
        return sum(distances_i)
