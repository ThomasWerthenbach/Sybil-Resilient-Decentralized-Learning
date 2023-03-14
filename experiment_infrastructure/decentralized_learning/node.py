from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from experiment_infrastructure.decentralized_learning.base_node import BaseNode
from experiment_infrastructure.experiment_settings.settings import Settings
from ml.aggregators.aggregator import Aggregator
from ml.models.model import Model
from ml.util import model_difference, model_sum


class Node(BaseNode):
    def __init__(self, model: Model, dataset: DataLoader, settings: Settings, node_id: int):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.settings = settings
        self.node_id = node_id

        self.models: Dict[int, Dict[int, nn.Module]] = defaultdict(dict)
        self.previous_models: Dict[int, nn.Module] = dict()

        self.accumulated_update_history: Dict[int, nn.Module] = dict()

        self.aggregator = Aggregator.get_aggregator_class(settings.aggregator)()

    def receive_model(self, model: nn.Module, peer: int, round: int):
        # Store model properly
        self.models[round][peer] = model

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

    def start_next_epoch(self, round: int) -> None:
        self.train(self.model, self.dataset, self.settings)
        self.receive_model(self.model, self.node_id, round)

    def aggregate(self, round: int) -> None:
        peers = list(self.models[round].keys())
        models = list(map(lambda p: self.models[round][p], peers))
        history = list(map(lambda p: self.accumulated_update_history[p], peers))
        self.model = self.aggregator.aggregate(models, history)
        self.previous_models = self.models[round]
        del self.models[round]

    def evaluate(self, test_data: DataLoader, attack_rate_data: DataLoader) -> Tuple[float, float]:
        with torch.no_grad():
            self.model.eval()
            test_loss = 0
            test_corr = 0
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device_name)
            self.model.to(device)
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

            test_att = 0.0
            if self.settings.sybil_attack:
                assert attack_rate_data is not None
                test_loss = 0
                test_corr = 0
                for data, target in attack_rate_data:
                    data, target = data.to(device), target.to(device)
                    output = self.model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    test_corr += pred.eq(target.view_as(pred)).sum().item()
                test_loss /= len(attack_rate_data)
                test_att = 100. * test_corr / (len(attack_rate_data) * 120)
                self.logger.info('Attack rate test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, test_corr, len(attack_rate_data) * 120, test_att))

            return test_acc, test_att

    def get_models(self) -> List[Model]:
        return [self.model]

    def get_ids(self) -> List[int]:
        return [self.node_id]
