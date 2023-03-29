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
        if name == 'FastMNIST':
            from ml.models.FasterMNIST import MNIST
            return MNIST
        elif name == 'MNIST':
            from ml.models.MNIST import MNIST
            return MNIST
        elif name == 'CIFAR10':
            from ml.models.CIFAR10LENET import LeNet
            return LeNet
        elif name == 'FashionMNIST':
            from ml.models.FashionMNIST import FashionMNISTCNN
            return FashionMNISTCNN
        raise NotImplementedError(f"No model found for {name}")
