from typing import Type

import torch
import torch.nn.functional as F
from torch import nn

from ml.datasets.EMNIST import EMNISTDataset
from ml.datasets.dataset import Dataset
from ml.models.model import Model

class EMNIST(Model):
    """
    Class for a LeNet Model for CIFAR10
    Inspired by original LeNet network for MNIST: https://ieeexplore.ieee.org/abstract/document/726791
    """

    def get_dataset_class(self) -> Type[Dataset]:
        return EMNISTDataset

    def __init__(self):
        """
        Constructor. Instantiates the CNN Model
            with 10 output classes
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.gn1 = nn.GroupNorm(2, 32)
        self.conv2 = nn.Conv2d(32, 32, 5, padding="same")
        self.gn2 = nn.GroupNorm(2, 32)
        self.conv3 = nn.Conv2d(32, 64, 5, padding="same")
        self.gn3 = nn.GroupNorm(2, 64)
        self.fc1 = nn.Linear(576, 62)

    def forward(self, x):
        """
        Forward pass of the model
        Parameters
        ----------
        x : torch.tensor
            The input torch tensor
        Returns
        -------
        torch.tensor
            The output torch tensor
        """
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = self.pool(F.relu(self.gn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class EMNISTOld(Model):
    """
    Adopted from https://nextjournal.com/gkoehler/pytorch-mnist
    """

    def get_dataset_class(self) -> Type[Dataset]:
        return EMNISTDataset

    def __init__(self):
        super(EMNISTOld, self).__init__()
        # self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding="same")
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding="same")
        # self.pool2 = nn.MaxPool2d(2, 2)
        # self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding="same")
        # self.pool3 = nn.MaxPool2d(2, 2)
        #
        # self.linear = nn.Linear(256 * 3 * 3, 1024)
        # self.linear2 = nn.Linear(1024, 62)

        self.linear = nn.Linear(784, 62)


    def forward(self, x):
        # x = x.view(-1, 1, 28, 28)
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool3(F.relu(self.conv3(x)))
        # x = x.view(-1, 256 * 3 * 3)
        # x = F.relu(self.linear(x))
        # x = self.linear2(x)
        x = x.view(-1, 784)
        x = self.linear(x)
        return F.log_softmax(x, dim=1)
