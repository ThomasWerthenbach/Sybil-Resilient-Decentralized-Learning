from abc import ABC, abstractmethod
from typing import Type

from torch import nn

from ml.datasets.dataset import Dataset


class Model(nn.Module, ABC):
    """
    Abstract class for a model
    """

    @abstractmethod
    def get_dataset_class(self) -> Type[Dataset]:
        pass

    @staticmethod
    def get_model_class(name: str) -> Type['Model']:
        if name == 'MNIST':
            from ml.models.FasterMNIST import MNIST
            return MNIST
        elif name == 'CIFAR10':
            from ml.models.CIFAR10LENET import LeNet
            return LeNet
        elif name == 'FashionMNIST':
            # FashionMNIST has the same dimensions as MNIST.
            from ml.models.FashionMNIST import FashionMNIST
            return FashionMNIST
        elif name == 'CelebA':
            from ml.models.Celeba import CelebA
            return CelebA
        raise NotImplementedError(f"No model found for {name}")
