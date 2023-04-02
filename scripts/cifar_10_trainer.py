import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from ml.models.CIFAR10CNN import ResNet32
from ml.models.CIFAR10LENET import LeNet


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
    fashion = DataLoader(torchvision.datasets.CIFAR10(
            root=os.path.join(os.path.dirname(__file__), '../data', 'train'), train=True, download=True, transform=ToTensor()
        ), batch_size=32, shuffle=True)
    fashion_test = DataLoader(torchvision.datasets.CIFAR10(
            root=os.path.join(os.path.dirname(__file__), '../data', 'test'), train=False, download=True, transform=ToTensor()
        ), batch_size=120, shuffle=True)

    # model = GNLeNet(input_channel=3, output=10, model_input=(32, 32))
    # model = ResNet32({"in_channels": 3, "out_channels": 10, "activation": "Default"})
    model = LeNet()

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(device_name)
    device = torch.device(device_name)
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.002)
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
            if i % 500 == 0:
                print(f"{i}/{len(fashion)} Loss: {loss.item()}")
        get_acc(model)
