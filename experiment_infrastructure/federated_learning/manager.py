import logging
from abc import abstractmethod

import torch
import torch.nn.functional as F
from ipv8.types import Peer
from torch.utils.data import DataLoader

from experiment_infrastructure.experiment_settings.settings import Settings
from ml.models.model import Model


class Manager:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def receive_model(self, peer_pk: Peer, info: bytes, model: bytes):
        pass

    @abstractmethod
    def start_next_epoch(self):
        ...

    @abstractmethod
    def get_settings(self) -> Settings:
        ...

    def train(self, model: Model, dataset: DataLoader) -> None:
        """
        Train the model for one epoch
        """
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        model = model.to(device)
        settings = self.get_settings()

        if dataset is None:
            raise RuntimeError("No peer dataset available")

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=settings.learning_rate,
            momentum=settings.momentum)

        model.train()
        train_loss = 0
        self.logger.info(f"Training for {settings.epochs} epochs with dataset length: {len(dataset)}")
        for _ in range(settings.epochs):
            for i, data in enumerate(dataset):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels)
                train_loss += F.nll_loss(outputs, labels, reduction='sum').item()
                loss.backward()
                optimizer.step()
