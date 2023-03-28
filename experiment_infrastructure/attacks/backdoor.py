from torch import Tensor
from torch.utils.data import DataLoader

from experiment_infrastructure.attacks.attack import Attack
from experiment_infrastructure.experiment_settings.settings import Settings
from ml.datasets.partitioner import Partition, DataPartitioner


class Backdoor(Attack):
    """
    Changes the very first point of every training sample to 1 and changes its target class
    """

    def __init__(self, settings: Settings, target_class: int = 1, seed=42):
        self.target_class = target_class
        self.seed = seed
        self.settings = settings

    def transform_data(self, data: Partition, trainset, sizes, peer_id) -> Partition:
        transformed_data = list()
        for i in range(len(data)):
            x, y = data[i]
            self.add_backdoor(x)
            transformed_data.append((x, self.target_class))

        return DataPartitioner(transformed_data, sizes).use(peer_id)

    def transform_eval_data(self, eval_data: DataLoader):
        raise NotImplementedError()

    def add_backdoor(self, tensor: Tensor):
        if self.settings.model == 'MNIST':
            tensor = tensor[0]
        # Add a square in the top left corner
        tensor[0][0].mul_(0)
        tensor[0][0].add_(1)
        tensor[0][1].mul_(0)
        tensor[0][1].add_(1)
        tensor[1][0].mul_(0)
        tensor[1][0].add_(1)
        tensor[1][1].mul_(0)
        tensor[1][1].add_(1)

