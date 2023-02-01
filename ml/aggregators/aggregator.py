from abc import abstractmethod
from typing import List

from torch import nn


class Aggregator:
    @abstractmethod
    def aggregate(self, delta: List[nn.Module], delta_history: List[nn.Module],
                  relevant_weights: List[int] = None) -> nn.Module:
        pass
