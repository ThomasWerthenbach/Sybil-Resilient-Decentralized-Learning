from torch.utils.data import DataLoader

from experiment_infrastructure.attacks.attack import Attack
from ml.datasets.partitioner import Partition


class Backdoor(Attack):
    """
    Changes the very first point of every training sample to 1 and changes its target class
    """

    def __init__(self, target_class: int = 1, seed=42):
        self.target_class = target_class
        self.seed = seed

    def transform_data(self, data: Partition, trainset, sizes, peer_id) -> Partition:
        transformed_data = list()
        for i in range(len(data)):
            x, y = data[i]
            first_tensor = x
            while first_tensor.shape:
                first_tensor = first_tensor[0]
            first_tensor.mul_(0)
            first_tensor.add_(1)
            transformed_data.append((x, self.target_class))

        return transformed_data

    def transform_eval_data(self, eval_data: DataLoader):
        transformed_data = list()
        for x, y in eval_data.dataset:
            first_tensor = x
            while first_tensor.shape:
                first_tensor = first_tensor[0]
            first_tensor.mul_(0)
            first_tensor.add_(1)
            transformed_data.append((x, self.target_class))

        return DataLoader(transformed_data, batch_size=120, shuffle=False)
