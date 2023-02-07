import logging
from abc import abstractmethod

import torch
import torch.nn.functional as F
from ipv8.types import Peer
from torch.utils.data import DataLoader

from experiment_settings.settings import Settings
from ml.models.model import Model


class Manager:
    @abstractmethod
    def receive_model(self, peer_pk: Peer, info: bytes, delta: bytes):
        pass

    @abstractmethod
    def start_next_epoch(self):
        pass

    @abstractmethod
    def get_dataset(self) -> DataLoader:
        pass

    @abstractmethod
    def get_settings(self) -> Settings:
        pass

    @abstractmethod
    def get_all_test_data(self) -> DataLoader:
        pass

    def train(self, model: Model):
        """
        Train the model for one epoch
        """
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        model = model.to(device)
        dataset = self.get_dataset()
        settings = self.get_settings()

        if dataset is None:
            raise RuntimeError("No peer dataset available")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=settings.learning_rate)

        model.train()
        train_loss = 0
        for i, data in enumerate(dataset, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
            train_loss += F.nll_loss(outputs, labels, reduction='sum').item()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                logging.info('Train Epoch status [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, len(dataset), 100. * i / len(dataset), loss.item()))

        # Evaluate the model on the full test set for results
        model.eval()
        test_dataset = self.get_all_test_data()
        test_loss = 0
        test_corr = 0
        with torch.no_grad():
            for data, target in test_dataset:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_corr += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_dataset)
        test_acc = 100. * test_corr / (len(test_dataset) * 120)
        logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, test_corr, len(test_dataset) * 120, test_acc))
