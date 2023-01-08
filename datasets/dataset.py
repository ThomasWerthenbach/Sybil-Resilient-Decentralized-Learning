from typing import Type

from ml.models.model import ModelType


class Dataset:
    DEFAULT_DATA_DIR = '../data'

    def load_trainset(self, batch_size=32, shuffle=False):
        """
        Function to load the training set
        """
        raise NotImplementedError()

    def load_testset(self, batch_size=32, shuffle=False):
        """
        Function to load the test set
        """
        raise NotImplementedError()

    @staticmethod
    def get_dataset_class(model: ModelType) -> Type['Dataset']:
        """
        Function to get the dataset class
        """
        if model == ModelType.MNIST:
            from datasets.MNIST import MNISTDataset
            return MNISTDataset
        elif model == ModelType.MNIST:
            from datasets.EMNIST import EMNISTLetterDataset
            return EMNISTLetterDataset
        elif model == ModelType.FashionMNIST:
            from datasets.FashionMNIST import FashionMNISTDataset
            return FashionMNISTDataset
        raise NotImplementedError()
