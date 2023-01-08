from typing import List

from torch import Tensor

from ml.prioritizers.prioritizer import Prioritizer


class AveragePrioritizer(Prioritizer):
    """
    The average prioritizer does not filter any models, as the average of all models will be used.
    """

    def filter_models(self, own_model: Tensor, other_models: List[Tensor]) -> List[Tensor]:
        return other_models
