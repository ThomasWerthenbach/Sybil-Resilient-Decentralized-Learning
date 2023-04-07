from typing import Type

import torch.nn.functional as F
from torch import nn

from ml.datasets.EMNIST import EMNISTDataset
from ml.datasets.dataset import Dataset
from ml.models.model import Model


class EMNIST(Model):
    """
    Adopted from https://nextjournal.com/gkoehler/pytorch-mnist
    """

    def get_dataset_class(self) -> Type[Dataset]:
        return EMNISTDataset

    def __init__(self):
        super(EMNIST, self).__init__()
        self.linear = nn.Linear(784, 62)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
