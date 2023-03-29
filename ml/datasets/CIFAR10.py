import logging

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from experiment_infrastructure.attacks.attack import Attack
from ml.datasets.dataset import Dataset
from ml.datasets.partitioner import DataPartitioner, DirichletDataPartitioner


class CIFAR10Dataset(Dataset):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def all_training_data(self, batch_size=32, shuffle=False) -> DataLoader:
        return DataLoader(torchvision.datasets.CIFAR10(
            root=self.DEFAULT_DATA_DIR + '/train', train=True, download=True, transform=ToTensor(),
        ), batch_size=batch_size, shuffle=shuffle)

    def get_peer_dataset(self, peer_id: int, total_peers: int, non_iid=False, sizes=None, batch_size=20, shuffle=False,
                         sybil_data_transformer: Attack = None):
        self.logger.info(f"Initializing dataset of size {1.0 / total_peers} for peer {peer_id}. Non-IID: {non_iid}")
        if sizes is None:
            sizes = [1.0 / total_peers for _ in range(total_peers)]
        data = torchvision.datasets.CIFAR10(
            root=self.DEFAULT_DATA_DIR + '/train', train=True, download=True, transform=ToTensor(),
        )
        if not non_iid:
            train_set = DataPartitioner(data, sizes).use(peer_id)
        else:
            train_set = DirichletDataPartitioner(
                data, sizes
            ).use(peer_id)
        if sybil_data_transformer is not None:
            train_data = {key: [] for key in range(10)}
            for x, y in data:
                train_data[y].append(x)
            train_set = sybil_data_transformer.transform_data(train_set, train_data, sizes, peer_id)

        return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

    def all_test_data(self, batch_size=32, shuffle=False):
        return DataLoader(torchvision.datasets.CIFAR10(
            root=self.DEFAULT_DATA_DIR + '/test', train=False, download=True, transform=ToTensor()
        ), batch_size=batch_size, shuffle=shuffle)
