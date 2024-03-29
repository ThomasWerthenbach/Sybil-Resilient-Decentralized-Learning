from abc import abstractmethod, ABC
from typing import List, Type

from torch import nn


class Aggregator(ABC):
    @abstractmethod
    def aggregate(self, own_model: nn.Module, own_history: nn.Module, models: List[nn.Module], history: List[nn.Module]) -> nn.Module:
        pass

    @staticmethod
    def get_aggregator_class(name: str) -> Type['Aggregator']:
        if name == 'average':
            from ml.aggregators.average import Average
            return Average
        elif name == 'foolsgold':
            from ml.aggregators.foolsgold import FoolsGold
            return FoolsGold
        elif name == 'sybilwall':
            from ml.aggregators.sybilwall import SybilWall
            return SybilWall
        elif name == 'sybilwall_weighted_median':
            from ml.aggregators.sybilwall_weighted_median import SybilWallWeightedMedian
            return SybilWallWeightedMedian
        elif name == 'sybilwall_krum':
            from ml.aggregators.sybilwall_krum_filter import SybilWallKrumFilter
            return SybilWallKrumFilter
        elif name == 'sybilwall_median':
            from ml.aggregators.sybilwall_median import SybilWallMedian
            return SybilWallMedian
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

    @staticmethod
    def flatten_one_layer(l: List) -> List:
        flat_list = []
        for sublist in l:
            if type(sublist) == list:
                for item in sublist:
                    flat_list.append(item)
            else:
                flat_list.append(sublist)
        return flat_list

    @staticmethod
    def flatten_all(parameters):
        for i in range(len(parameters)):
            while len(parameters[i]) > 0 and any(map(lambda x: type(x) == list, parameters[i])):
                parameters[i] = Aggregator.flatten_one_layer(parameters[i])
