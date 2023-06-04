from typing import Type

from ml.datasets.FashionMNIST import FashionMNISTDataset
from ml.datasets.dataset import Dataset
from ml.models.MNIST import MNIST


class FashionMNIST(MNIST):
    def get_dataset_class(self) -> Type[Dataset]:
        return FashionMNISTDataset
