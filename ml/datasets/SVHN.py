import logging

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from experiment_infrastructure.attacks.attack import Attack
from ml.datasets.dataset import Dataset


class SVHNDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def all_training_data(self, batch_size=32, shuffle=False) -> DataLoader:
        return DataLoader(torchvision.datasets.SVHN(
            root=self.DEFAULT_DATA_DIR + '/train', split='train', download=True, transform=ToTensor(),
        ), batch_size=batch_size, shuffle=shuffle)

    def get_peer_dataset(self, peer_id: int, total_peers: int, non_iid=False, sizes=None, batch_size=8, alpha=0.1, shuffle=True,
                         sybil_data_transformer: Attack = None) -> DataLoader:
        data = torchvision.datasets.SVHN(
            root=self.DEFAULT_DATA_DIR + '/train', split='train', download=True, transform=ToTensor(),
        )
        return self.get_peer_train_set(data, peer_id, total_peers, non_iid, sizes, batch_size, alpha, shuffle,
                                       sybil_data_transformer)

    def all_test_data(self, batch_size=32, shuffle=False):
        return DataLoader(torchvision.datasets.SVHN(
            root=self.DEFAULT_DATA_DIR + '/test', split='test', download=True, transform=ToTensor()
        ), batch_size=batch_size, shuffle=shuffle)
