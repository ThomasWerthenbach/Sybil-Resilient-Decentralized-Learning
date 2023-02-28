import os
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from experiment_infrastructure.attacks.attack import Attack


class Dataset(ABC):
    NUM_CLASSES = None
    DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')

    @abstractmethod
    def all_training_data(self, batch_size=32, shuffle=False) -> DataLoader:
        """
        Retrieves the entire dataset (used for transfer learning preparations)
        """

    @abstractmethod
    def get_peer_dataset(self, peer_id: int, total_peers: int, non_iid, sizes=None,
                         sybil_data_transformer: Attack = None) -> DataLoader:
        """
        Function to load the training set
        """

    @abstractmethod
    def all_test_data(self, batch_size=32, shuffle=False) -> DataLoader:
        """
        Function to load the test set
        """
