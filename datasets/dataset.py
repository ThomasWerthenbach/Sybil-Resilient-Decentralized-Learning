from typing import Type


class Dataset:
    DEFAULT_DATA_DIR = '../data'

    def load_trainset(self, batch_size=32, shuffle=False):
        """
        Function to load the training set
        """
        raise NotImplementedError

    def load_testset(self, batch_size=32, shuffle=False):
        """
        Function to load the test set
        """
        raise NotImplementedError

    @staticmethod
    def get_dataset_class(dataset: str) -> Type['Dataset']:
        """
        Function to get the dataset class
        """
        if dataset.lower() == 'mnist':
            from datasets.MNIST import MNISTDataset
            return MNISTDataset
        elif dataset.lower() == 'emnist':
            from datasets.EMNIST import EMNISTLetterDataset
            return EMNISTLetterDataset
        elif dataset.lower() == 'fashionmnist':
            from datasets.FashionMNIST import FashionMNISTDataset
            return FashionMNISTDataset
        raise NotImplementedError
