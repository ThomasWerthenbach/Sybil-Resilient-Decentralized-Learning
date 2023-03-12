from torch.utils.data import DataLoader

from experiment_infrastructure.attacks.attack import Attack
from ml.datasets.partitioner import Partition, DataPartitioner


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

        return DataPartitioner(transformed_data, sizes).use(peer_id)

    def transform_eval_data(self, eval_data: DataLoader):
        raise NotImplementedError()
