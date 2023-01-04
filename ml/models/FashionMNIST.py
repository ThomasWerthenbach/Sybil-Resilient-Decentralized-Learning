import torch
from torch import nn
import torch.nn.functional as F

from ml.models.model import Model


class FashionMNISTCNN(Model):
    """
    Class for a CNN Model for FashionMNIST
    Inspired by: doi.org/10.1109/ITCE48509.2020.9047776
    (And Martijn's framework)
    """
    # todo WIP

    def __init__(self):
        super().__init__()

        # self.lenet = nn.Sequential()
        # self.lenet.add_module("conv1", nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2))
        # self.lenet.add_module("tanh1", nn.Tanh())
        # self.lenet.add_module("avg_pool1", nn.AvgPool2d(kernel_size=2, stride=2))
        # self.lenet.add_module("conv2", nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1))
        # self.lenet.add_module("avg_pool2", nn.AvgPool2d(kernel_size=2, stride=2))
        # self.lenet.add_module("conv3", nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5,stride=1))
        # self.lenet.add_module("tanh2", nn.Tanh())
        # self.lenet.add_module("flatten", nn.Flatten(start_dim=1))
        # lenet.add_module("fc1", nn.Linear(in_features=120 , out_features=84))
        # lenet.add_module("tanh3", nn.Tanh())
        # lenet.add_module("fc2", nn.Linear(in_features=84, out_features=10))

        # self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        # self.avg1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        # self.avg2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        # self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))

        # Martijn's:
        # 1.6 million params
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # x = self.conv1.forward(x)
        # x = self.avg1.forward(x)
        # x = self.conv2.forward(x)
        # x = self.avg2.forward(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1.forward(x)
        # x = self.fc2.forward(x)
        # x = self.fc3.forward(x)
        # return x

        # Martijn's:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
