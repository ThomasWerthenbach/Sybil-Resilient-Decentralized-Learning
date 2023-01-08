from typing import List

import torch
from torch import Tensor

from ml.prioritizers.prioritizer import Prioritizer


class BristleSettings:
    alpha: float = 0.4
    min_number_models_for_distance_screening: int = 10


def get_distances(own_model: Tensor, models: List[Tensor]) -> List[Tensor]:
    dist = [(model, torch.dist(own_model, model)) for model in models]
    dist.sort(key=lambda x: float(x[1][0]))
    return list(map(lambda x: x[0], dist))


class BristlePrioritizer(Prioritizer):
    """
    Bristle uses a distance-based prioritizer to filter for the most similar models.
    """

    def __init__(self, settings: BristleSettings):
        self.settings = settings

    def filter_models(self, own_model: Tensor, other_models: List[Tensor]) -> List[Tensor]:
        if len(other_models) < self.settings.min_number_models_for_distance_screening:
            return other_models

        distances = get_distances(own_model, other_models)

        spl = list()
        spl.append(distances[:len(distances) // 3])
        spl.append(distances[len(distances) // 3:2 * len(distances) // 3])
        spl.append(distances[2 * len(distances) // 3:])

        fl = int(round(((1.0 - self.settings.alpha) ** 2) * self.settings.min_number_models_for_distance_screening))
        fm = int(round((-2 * self.settings.alpha ** 2 + 2 * self.settings.alpha))
                 * self.settings.min_number_models_for_distance_screening)
        fh = int(round((self.settings.alpha ** 2) * self.settings.min_number_models_for_distance_screening))

        res = list()
        res += spl[0][:fl]
        res += spl[1][:fm]
        res += spl[2][:fh]

        return list(map(lambda x: x[0], res))
