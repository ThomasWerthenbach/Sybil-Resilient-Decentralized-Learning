from typing import Type

import torch
from torch import nn
import torch.nn.functional as F

from ml.datasets.FashionMNIST import FashionMNISTDataset
from ml.datasets.dataset import Dataset
from ml.models.model import Model


class FashionMNISTCNN(Model):
    """
    Class for a CNN Model for FashionMNIST
    Based on: doi.org/10.1109/ITCE48509.2020.9047776
    """

    def get_dataset_class(self) -> Type[Dataset]:
        return FashionMNISTDataset

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5), stride=1, padding=2)  # C1
        self.avg1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)  # S2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=1, padding=0)  # C3
        self.avg2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)  # S4
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5), stride=1, padding=0)  # C5
        self.fc1 = nn.Linear(120, 84)  # F6
        self.fc2 = nn.Linear(84, 10)  # Output layer

    def forward(self, x):
        x = F.relu(self.conv1.forward(x))
        x = self.avg1.forward(x)
        x = F.relu(self.conv2.forward(x))
        x = self.avg2.forward(x)
        x = F.relu(self.conv3.forward(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1.forward(x))
        x = self.fc2.forward(x)
        return x


# class FashionCNN(nn.Module):
#     """
#     no paper :(
#     https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy/notebook
#     """
#
#     def __init__(self):
#         super(FashionCNN, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
#         self.drop = nn.Dropout2d(0.25)
#         self.fc2 = nn.Linear(in_features=600, out_features=120)
#         self.fc3 = nn.Linear(in_features=120, out_features=10)
#
#     def forward(self, x):
#         out = self.layer1(x)
#         out = self.layer2(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc1(out)
#         out = self.drop(out)
#         out = self.fc2(out)
#         out = self.fc3(out)
#
#         return out
