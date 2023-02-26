from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.decentralized_learning.base_node import BaseNode
from experiments.experiment_settings.settings import Settings
from ml.aggregators.aggregator import Aggregator
from ml.models.model import Model
from ml.util import model_difference, model_sum


class Node(BaseNode):
    def __init__(self, model: Model, dataset: DataLoader, settings: Settings):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.settings = settings

        self.models: Dict[int, nn.Module] = dict()
        self.previous_models: Dict[int, nn.Module] = dict()

        self.accumulated_update_history: Dict[int, nn.Module] = dict()

        self.aggregator = Aggregator.get_aggregator_class(settings.aggregator)()

    def receive_model(self, model: nn.Module, peer: int):
        # Store model properly
        self.models[peer] = model

        # Calculate difference with previous model
        if peer in self.previous_models:
            difference = model_difference(self.previous_models[peer], model)
        else:
            difference = model

        # Store model updates properly
        if peer in self.accumulated_update_history:
            self.accumulated_update_history[peer] = model_sum(self.accumulated_update_history[peer], difference)
        else:
            self.accumulated_update_history[peer] = difference

    def start_next_epoch(self) -> Model:
        return self.train(self.model, self.dataset, self.settings)

    def aggregate(self) -> None:
        peers = list(self.models.keys())
        models = list(map(lambda p: self.models[p], peers))
        history = list(map(lambda p: self.accumulated_update_history[p], peers))
        self.model = self.aggregator.aggregate(models, history)
        self.previous_models = self.models
        self.models = dict()

    def evaluate(self, test_data: DataLoader) -> float:
        self.model.eval()
        test_loss = 0
        test_corr = 0
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        self.model.to(device)
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_corr += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_data)
        test_acc = 100. * test_corr / (len(test_data) * 120)
        self.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, test_corr, len(test_data) * 120, test_acc))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return test_acc

