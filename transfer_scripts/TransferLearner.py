import logging

import numpy as np
import torch
import torch.nn.functional as F

from datasets.dataset import Dataset
from ml.models.model import Model
from ml.transfer_settings import TransferSettings


logging.basicConfig(level=logging.INFO)

class TransferLearner:
    def __init__(self, settings: TransferSettings):
        self.settings = settings

    def train(self):
        dataset = Dataset.get_dataset_class(self.settings.dataset)()
        trainloader = dataset.load_trainset(batch_size=self.settings.train_batch_size, shuffle=self.settings.train_shuffle)
        testloader = dataset.load_testset(batch_size=self.settings.test_batch_size, shuffle=self.settings.test_shuffle)

        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)

        model = Model.get_model_class(self.settings.model)().to(device)

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.settings.learning_rate)

        max_correct = 0
        for epoch in range(self.settings.epochs):
            model.train()
            train_loss = 0
            train_corr = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels)
                train_pred = outputs.argmax(dim=1, keepdim=True)
                train_corr += train_pred.eq(labels.view_as(train_pred)).sum().item()
                train_loss += F.nll_loss(outputs, labels, reduction='sum').item()
                loss.backward()
                optimizer.step()
                if i % 50 == 0:
                    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i, len(trainloader),
                               100. * i / len(trainloader), loss.item()))
            model.eval()
            test_loss = 0
            correct = 0
            total_pred = np.zeros(0)
            total_target = np.zeros(0)
            with torch.no_grad():
                for data, target in testloader:
                    data, target = data.to(device), target.to(device, dtype=torch.int64)
                    output = model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    total_pred = np.append(total_pred, pred.cpu().numpy())
                    total_target = np.append(total_target, target.cpu().numpy())
                    correct += pred.eq(target.view_as(pred)).sum().item()
                if max_correct < correct:
                    torch.save(model.state_dict(), "mnist-10-epoch.pth")
                    max_correct = correct
                    logging.info("Best accuracy! correct images: %5f" % (correct / (len(testloader) * len(target))))
