from abc import ABC, abstractmethod
from typing import Type

from torch import nn

from datasets.dataset import Dataset
from ml.models.MNIST import MNIST


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
            return MNIST
        raise NotImplementedError(f"No model found for {name}")
