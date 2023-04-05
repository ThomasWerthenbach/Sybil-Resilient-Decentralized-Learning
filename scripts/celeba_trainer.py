import os

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from ml.datasets.CelebA import Celeba
from ml.datasets.FEMNIST import FEMNIST
from ml.models.Celeba import CNN
from ml.models.MNIST import MNIST

def get_acc(model):
    with torch.no_grad():
        model.eval()
        test_corr = 0
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        model.to(device)
        for data, target in test:
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_corr += pred.eq(target.view_as(pred)).sum().item()
        test_acc = 100. * test_corr / (len(test) * 120)
        print(test_acc)

if __name__ == '__main__':
    train = Celeba().all_training_data()
    test = Celeba().all_test_data()

    model = CNN()

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(device_name)
    device = torch.device(device_name)
    model = model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001)
    error = nn.CrossEntropyLoss()

    print(f"Training for {3} epochs with dataset length: {len(train)}")
    for _ in range(50):
        model.train()
        for i, data in enumerate(train):
            inputs, labels = data
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = error(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"{i}/{len(train)} Loss: {loss.item()}")
                # break
        get_acc(model)
