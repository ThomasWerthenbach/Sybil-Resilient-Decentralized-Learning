from abc import abstractmethod
from typing import List

from torch import nn


class Aggregator:
    @abstractmethod
    def aggregate(self, models: List[nn.Module], relevant_weights: List[int] = None):
        pass
