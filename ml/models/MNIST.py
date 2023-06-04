from typing import Type

import torch.nn.functional as F
from torch import nn

from ml.datasets.MNIST import MNISTDataset
from ml.datasets.dataset import Dataset
from ml.models.model import Model


class MNIST(Model):
    """
    Adopted from https://nextjournal.com/gkoehler/pytorch-mnist
    """

    def get_dataset_class(self) -> Type[Dataset]:
        return MNISTDataset

    def __init__(self):
        super(MNIST, self).__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
