from abc import ABC, abstractmethod
from typing import List

from torch import Tensor


class Prioritizer(ABC):
    @abstractmethod
    def filter_models(self, own_model: Tensor, other_models: List[Tensor]) -> List[Tensor]:
        pass
