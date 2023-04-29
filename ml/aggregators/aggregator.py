from abc import abstractmethod, ABC
from typing import List, Type

from torch import nn


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module], delta_history: List[nn.Module],
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
        elif name == 'repple':
            from ml.aggregators.repple import Repple
            return Repple
        elif name == 'repple2':
            from ml.aggregators.repple2 import Repple2
            return Repple2
        elif name == 'median':
            from ml.aggregators.median import Median
            return Median
        elif name == 'krum':
            from ml.aggregators.krum import Krum
            return Krum
        elif name == 'multi-krum':
            from ml.aggregators.multi_krum import MultiKrum
            return MultiKrum
        raise NotImplementedError(f"No aggregator found for {name}")

    @abstractmethod
    def requires_gossip(self) -> bool:
        ...

    def flatten_one_layer(self, l: List) -> List:
        flat_list = []
        for sublist in l:
            if type(sublist) == list:
                for item in sublist:
                    flat_list.append(item)
            else:
                flat_list.append(sublist)
        return flat_list

    def flatten_all(self, parameters):
        for i in range(len(parameters)):
            while len(parameters[i]) > 0 and any(map(lambda x: type(x) == list, parameters[i])):
                parameters[i] = self.flatten_one_layer(parameters[i])
