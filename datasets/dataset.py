from abc import ABC, abstractmethod
from typing import Type

from ml.models.model import ModelType


class Dataset(ABC):
    NUM_CLASSES = None
    DEFAULT_DATA_DIR = '../data'

    @abstractmethod
    def all_training_data(self, batch_size=32, shuffle=False):
        """
        Retrieves the entire dataset (used for transfer learning preparations)
        """

    @abstractmethod
    def get_peer_dataset(self, peer_id: int, total_peers: int, non_iid, sizes=None):
        """
        Function to load the training set
        """

    @abstractmethod
    def all_test_data(self, batch_size=32, shuffle=False):
        """
        Function to load the test set
        """

    @staticmethod
    def get_dataset_class(model: ModelType) -> Type['Dataset']:
        """
        Function to get the dataset class
        """
        if model == ModelType.MNIST:
            from datasets.MNIST import MNISTDataset
            return MNISTDataset
        elif model == ModelType.EMNIST:
            from datasets.EMNIST import EMNISTLetterDataset
            return EMNISTLetterDataset
        elif model == ModelType.FashionMNIST:
            from datasets.FashionMNIST import FashionMNISTDataset
            return FashionMNISTDataset
        raise NotImplementedError()
