from abc import ABC, abstractmethod
from typing import Type

from ml.datasets.partitioner import Partition


class Attack(ABC):
    @abstractmethod
    def transform_data(self, data: Partition, trainset, sizes, peer_id) -> Partition:
        ...

    @staticmethod
    def get_attack_class(attack_name: str) -> Type['Attack']:
        if attack_name == 'label_flip':
            from experiment_infrastructure.attacks.label_flip import LabelFlip
            return LabelFlip
        else:
            raise NotImplementedError(f"Unknown attack name {attack_name}")