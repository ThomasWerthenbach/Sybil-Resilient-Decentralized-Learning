from abc import ABC, abstractmethod
from typing import Type

from torch.utils.data import DataLoader

from ml.datasets.partitioner import Partition


class Attack(ABC):
    @abstractmethod
    def transform_data(self, data: Partition, trainset, sizes, peer_id) -> Partition:
        ...

    @abstractmethod
    def transform_eval_data(self, eval_data: DataLoader):
        ...

    @staticmethod
    def get_attack_class(attack_name: str) -> Type['Attack']:
        if attack_name == 'label_flip':
            from experiment_infrastructure.attacks.label_flip import LabelFlip
            return LabelFlip
        elif attack_name == 'backdoor':
            from experiment_infrastructure.attacks.backdoor import Backdoor
            return Backdoor
        else:
            raise NotImplementedError(f"Unknown attack name {attack_name}")
