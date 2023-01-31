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
    def receive_model(self, peer_pk: Peer, model: bytes):
        pass

    @abstractmethod
    def start_next_epoch(self):
        pass

    @abstractmethod
    def get_dataset(self) -> DataLoader:
        pass

    @abstractmethod
    def get_model(self) -> Model:
        pass

    @abstractmethod
    def get_settings(self) -> Settings:
        pass

    def train(self):
        """
        Train the model for one epoch
        """
        dataset = self.get_dataset()
        model = self.get_model()
        settings = self.get_settings()

        if dataset is None:
            raise RuntimeError("No peer dataset available")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=settings.learning_rate)

        model.train()
        train_loss = 0
        train_corr = 0
        for i, data in enumerate(dataset, 0):
            inputs, labels = data
            inputs, labels = inputs.to('cpu'), labels.to('cpu')
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
            train_pred = outputs.argmax(dim=1, keepdim=True)
            train_corr += train_pred.eq(labels.view_as(train_pred)).sum().item()
            train_loss += F.nll_loss(outputs, labels, reduction='sum').item()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                logging.info('Train Epoch status [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, len(dataset), 100. * i / len(dataset), loss.item()))
