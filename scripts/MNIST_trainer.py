import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from ml.models.MNIST import MNIST


# class Network(nn.Module):
#   def __init__(self):
#     super().__init__()
#
#     # define layers
#     self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
#     self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
#
#     self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
#     self.fc2 = nn.Linear(in_features=120, out_features=60)
#     self.out = nn.Linear(in_features=60, out_features=10)
#
#   # define forward function
#   def forward(self, t):
#     # conv 1
#     t = self.conv1(t)
#     t = F.relu(t)
#     t = F.max_pool2d(t, kernel_size=2, stride=2)
#
#     # conv 2
#     t = self.conv2(t)
#     t = F.relu(t)
#     t = F.max_pool2d(t, kernel_size=2, stride=2)
#
#     # fc1
#     t = t.reshape(-1, 12*4*4)
#     t = self.fc1(t)
#     t = F.relu(t)
#
#     # fc2
#     t = self.fc2(t)
#     t = F.relu(t)
#
#     # output
#     t = self.out(t)
#     # don't need softmax here since we'll use cross-entropy as activation.
#
#     return t

def get_acc(model):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        test_corr = 0
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        model.to(device)
        for data, target in fashion_test:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            test_corr += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(fashion_test)
        test_acc = 100. * test_corr / (len(fashion_test) * 120)
        print(test_acc)

if __name__ == '__main__':
    fashion = DataLoader(torchvision.datasets.MNIST(
            root=os.path.join(os.path.dirname(__file__), '../data', 'train'), train=True, download=True, transform=ToTensor()
        ), batch_size=32, shuffle=True)
    fashion_test = DataLoader(torchvision.datasets.MNIST(
            root=os.path.join(os.path.dirname(__file__), '../data', 'test'), train=False, download=True, transform=ToTensor()
        ), batch_size=120, shuffle=True)

    model = MNIST()

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(device_name)
    device = torch.device(device_name)
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001)
    error = nn.CrossEntropyLoss()

    print(f"Training for {3} epochs with dataset length: {len(fashion)}")
    for _ in range(50):
        model.train()
        for i, data in enumerate(fashion):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = error(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"{i}/{len(fashion)} Loss: {loss.item()}")
                # break
        get_acc(model)
