from abc import ABC, abstractmethod
from typing import List

from torch import Tensor


class Integrator(ABC):
    @abstractmethod
    def integrate(self, own_model: Tensor, other_models: List[Tensor]) -> Tensor:
        pass
