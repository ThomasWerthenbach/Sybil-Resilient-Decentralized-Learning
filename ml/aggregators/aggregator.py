from abc import abstractmethod, ABC
from typing import List, Type

from torch import nn


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, models: List[nn.Module], delta_history: List[nn.Module],
                  relevant_weights: List[int] = None) -> nn.Module:
        pass

    @staticmethod
    def get_aggregator_class(name: str) -> Type['Aggregator']:
        if name == 'average':
            from ml.aggregators.average import Average
            return Average
        elif name == 'foolsgold':
            from ml.aggregators.foolsgold import FoolsGold
            return FoolsGold
        raise NotImplementedError(f"No aggregator found for {name}")
