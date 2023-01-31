from abc import ABC, abstractmethod
from typing import Type

from torch import nn

from datasets.dataset import Dataset


class Model(nn.Module, ABC):
    """
    Abstract class for a model
    """

    @abstractmethod
    def get_dataset_class(self) -> Type[Dataset]:
        pass
