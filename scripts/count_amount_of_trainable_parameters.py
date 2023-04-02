import numpy as np
from torch import nn

from ml.models.CIFAR10CNN import ResNet8, ResNet14, ResNet56, ResNet44, ResNet20, ResNet26, ResNet32
from ml.models.CIFAR10LENET import LeNet
from ml.models.FasterMNIST import MNIST
from ml.models.model import Model
import torch.nn.functional as F
import torch.optim as optim

from ml.util import serialize_model
import torch
import torch.nn.functional as F
from torch import nn

class CNN(nn.Module):
    """
    Class for a CNN Model for Celeba

    """

    def __init__(self):
        """
        Constructor. Instantiates the CNN Model
            with 84*84*3 Input and 2 output classes

        """
        super().__init__()
        # 2.8k parameters
        self.conv1 = nn.Conv2d(3, 32, 3, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding="same")
        self.conv3 = nn.Conv2d(32, 32, 3, padding="same")
        self.conv4 = nn.Conv2d(32, 32, 3, padding="same")
        self.fc1 = nn.Linear(5 * 5 * 32, 10)

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
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


if __name__ == '__main__':

    for model in [CNN(), LeNet()]:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        print(sum([np.prod(p.size()) for p in model_parameters]))
