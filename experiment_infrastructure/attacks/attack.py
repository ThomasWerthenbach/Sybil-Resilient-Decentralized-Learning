from abc import ABC, abstractmethod

from ml.datasets.partitioner import Partition


class Attack(ABC):
    @abstractmethod
    def transform_data(self, data: Partition) -> Partition:
        ...