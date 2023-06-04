from typing import Type

from ml.datasets.SVHN import SVHNDataset
from ml.datasets.dataset import Dataset
from ml.models import CIFAR10


class LeNet(CIFAR10.LeNet):
    def get_dataset_class(self) -> Type[Dataset]:
        return SVHNDataset
