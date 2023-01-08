import copy
from typing import List

from torch import Tensor

from ml.integrators.integrator import Integrator


class AverageIntegrator(Integrator):
    """
    The average integrator takes the average of all models including its own and returns this as the new model.
    """
    def integrate(self, own_model: Tensor, other_models: List[Tensor]) -> Tensor:
        weight = 1 / (len(other_models) + 1)
        center_model = copy.deepcopy(own_model)
        center_model.mul_(weight)  # account for own model
        for m in other_models:
            center_model.add_(m * weight)
        return center_model
