import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiment_infrastructure.experiment_settings.settings import Settings
from ml.models.model import Model
import torch.nn.functional as F


class BaseNode(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def receive_model(self, model: nn.Module, peer: int, round: int) -> None:
        ...

    @abstractmethod
    def receive_distant_model(self, model: nn.Module, peer: int, round: int, distance: int, distant_peer: int) -> None:
        ...

    @abstractmethod
    def get_models(self) -> List[Model]:
        ...

    @abstractmethod
    def start_next_epoch(self, round: int) -> None:
        ...

    @abstractmethod
    def get_ids(self) -> List[int]:
        ...

    @abstractmethod
    def aggregate(self, round: int) -> None:
        ...

    @abstractmethod
    def evaluate(self, test_data: DataLoader, attack_rate: DataLoader) -> Tuple[float, float]:
        ...

    @abstractmethod
    def get_random_neighbour_history(self, for_peer: int) -> Tuple[int, int, int, nn.Module] | None:
        ...

    def train(self, model: Model, dataset: DataLoader, settings: Settings):
        """
        Train the model for one epoch
        """
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        model = model.to(device)

        if dataset is None:
            raise RuntimeError("No peer dataset available")

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=settings.learning_rate,
            momentum=settings.momentum)

        model.train()
        self.logger.info(f"Training for {settings.epochs} epochs with dataset length: {len(dataset)}")
        for _ in range(settings.epochs):
            for i, data in enumerate(dataset):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels)
                loss.backward()
                optimizer.step()
