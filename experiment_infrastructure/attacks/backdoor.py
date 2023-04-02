from typing import List

from torch import Tensor
from torch.utils.data import DataLoader

from experiment_infrastructure.attacks.attack import Attack
from experiment_infrastructure.experiment_settings.settings import Settings
from ml.datasets.partitioner import Partition


class Backdoor(Attack):
    """
    Changes the very first point of every training sample to 1 and changes its target class
    """

    def __init__(self, settings: Settings, target_class: int = 1, seed=42):
        self.target_class = target_class
        self.seed = seed
        self.settings = settings

    def transform_data(self, data: Partition, trainset, sizes, peer_id) -> List:
        transformed_data = list()
        for x, y in data:
            self.add_backdoor(x)
            transformed_data.append((x, self.target_class))

        return transformed_data

    def transform_eval_data(self, eval_data: DataLoader):
        transformed_data = list()
        for x, y in eval_data.dataset:
            # To calculate the attack rate, we should only retrieve samples from another class than the target class
            if y != self.target_class:
                self.add_backdoor(x)
                transformed_data.append((x, self.target_class))

        return DataLoader(transformed_data, batch_size=120, shuffle=False)

    def change_value_to_1(self, tensor: Tensor):
        tensor.mul_(0)
        tensor.add_(1)

    def add_backdoor(self, tensor: Tensor):
        if self.settings.model == 'MNIST':
            tensor = tensor[0]
        if self.settings.model == 'FashionMNIST':
            tensor = tensor[0]
        # Add a square in the top left corner
        self.change_value_to_1(tensor[0][0])
        self.change_value_to_1(tensor[0][1])
        self.change_value_to_1(tensor[0][2])
        self.change_value_to_1(tensor[1][0])
        self.change_value_to_1(tensor[1][1])
        self.change_value_to_1(tensor[1][2])
        self.change_value_to_1(tensor[2][0])
        self.change_value_to_1(tensor[2][1])
        self.change_value_to_1(tensor[2][2])
