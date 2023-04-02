from typing import Type

import torch.nn.functional as F
from torch import nn

from ml.datasets.KDDCup99 import KDDCup99Dataset
from ml.datasets.MNIST import MNISTDataset
from ml.datasets.dataset import Dataset


class KDDCup99(nn.Module):
    """
    Adopted from https://nextjournal.com/gkoehler/pytorch-mnist
    """

    def get_dataset_class(self) -> Type[Dataset]:
        return KDDCup99Dataset

    def __init__(self):
        super(KDDCup99, self).__init__()
        self.linear = nn.Linear(41, 5)
        # self.linear = nn.Linear(41, 23)
        # self.linear2 = nn.Linear(23, 5)

    def forward(self, x):
        x = self.linear(x)
        # x = self.linear2(x)
        return F.softmax(x, dim=1)
