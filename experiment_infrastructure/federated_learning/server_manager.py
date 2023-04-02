import json
from collections import defaultdict
from functools import reduce
from operator import add
from typing import Dict, Callable, List

import torch
import torch.nn.functional as F
from ipv8.types import Peer
from torch import nn

from experiment_infrastructure.attacks.attack import Attack
from experiment_infrastructure.experiment_settings.settings import Settings
from experiment_infrastructure.federated_learning.manager import Manager
from ml.aggregators.aggregator import Aggregator
from ml.models.model import Model
from ml.util import deserialize_model, serialize_model, model_difference, model_sum


class ServerManager(Manager):
    """
    1. Check if all nodes have submitted a model
    2. Aggregate
    3. Send aggregated model to all nodes
    """

    def __init__(self, settings: Settings, send_model: Callable[[Peer, bytes, bytes], None], statistic_logger: Callable[[str, float], None]):
        super().__init__()
        self.send_model = send_model
        self.round = 0
        self.statistic_logger = statistic_logger
        self.expecting_models = settings.total_hosts * settings.peers_per_host + settings.sybil_amount
        self.rounds: Dict[int, Dict[Peer, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.previous_models: Dict[Peer, Dict[int, nn.Module]] = defaultdict(dict)
        self.models: Dict[Peer, Dict[int, nn.Module]] = defaultdict(dict)
        self.accumulated_update_history: Dict[Peer, Dict[int, nn.Module]] = defaultdict(dict)
        self.settings = settings
        self.data = Model.get_model_class(settings.model)().get_dataset_class()().all_test_data(120)
        self.aggregator: Aggregator = Aggregator.get_aggregator_class(settings.aggregator)()

        if settings.sybil_attack:
            attack = Attack.get_attack_class(settings.sybil_attack_type)(settings)
            self.attack_rate_data = attack.transform_eval_data(self.data)

    def receive_model(self, host: Peer, info: bytes, serialized_model: bytes):
        info = json.loads(info.decode())
        r = info['round']
        p = info['peer']
        if self.round != r:
            self.logger.info("Server received model with incorrect round number")
            return
        self.logger.info("Server received model")

        if p in self.rounds[r][host]:
            self.logger.info("Server received model from peer twice")
            return
        self.rounds[r][host].append(p)

        self.expecting_models -= 1

        # Store model properly
        model = deserialize_model(serialized_model, self.settings)
        self.models[host][p] = model

        # Calculate model difference
        if p in self.previous_models[host]:
            difference = model_difference(self.previous_models[host][p], model)
        else:
            difference = model

        # Store model updates properly
        if p in self.accumulated_update_history[host]:
            self.accumulated_update_history[host][p] = model_sum(self.accumulated_update_history[host][p], difference)
        else:
            self.accumulated_update_history[host][p] = difference

        # Perform aggregation if all models are received
        if self.expecting_models == 0:
            self.expecting_models = self.settings.total_hosts * self.settings.peers_per_host + self.settings.sybil_amount
            peers = list(self.models.keys())
            models = reduce(add, map(lambda x: list(self.models[x].values()), peers))
            history = reduce(add, map(lambda x: list(self.accumulated_update_history[x].values()), peers))

            result = self.aggregator.aggregate(models, history)
            self.previous_models = self.models
            self.models = defaultdict(dict)

            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device_name)

            result.eval()
            test_loss = 0
            test_corr = 0
            result.to(device)
            with torch.no_grad():
                for data, target in self.data:
                    data, target = data.to(device), target.to(device)
                    output = result(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    test_corr += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(self.data)
            test_acc = 100. * test_corr / (len(self.data) * 120)
            self.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, test_corr, len(self.data) * 120, test_acc))
            self.statistic_logger("accuracy", test_acc)

            if self.settings.sybil_attack:
                test_loss = 0
                test_corr = 0
                with torch.no_grad():
                    for data, target in self.attack_rate_data:
                        data, target = data.to(device), target.to(device)
                        output = result(data)
                        test_loss += F.nll_loss(output, target, reduction='sum').item()
                        pred = output.argmax(dim=1, keepdim=True)
                        test_corr += pred.eq(target.view_as(pred)).sum().item()
                test_loss /= len(self.attack_rate_data)
                test_acc = 100. * test_corr / (len(self.attack_rate_data) * 120)
                self.logger.info('Attack rate test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, test_corr, len(self.attack_rate_data) * 120, test_acc))
                self.statistic_logger("attack_rate", test_acc)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Send aggregated delta to all nodes
            self.logger.info("Server finished aggregation")
            self.round += 1
            for host in peers:
                self.send_model(host, json.dumps({'round': self.round}).encode(), serialize_model(result))

    def start_next_epoch(self):
        pass

    def get_settings(self) -> Settings:
        return self.settings
