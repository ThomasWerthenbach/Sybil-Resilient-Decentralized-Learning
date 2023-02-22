import json
from collections import defaultdict
from functools import reduce
from operator import add
from typing import Dict, Callable, List

import torch
import torch.nn.functional as F
from ipv8.types import Peer
from torch import nn
from torch.utils.data import DataLoader

from experiment_settings.settings import Settings
from federated_learning.manager import Manager
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
        self.expecting_models = settings.total_hosts * settings.peers_per_host
        self.rounds: Dict[int, Dict[Peer, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.models: Dict[Peer, Dict[int, nn.Module]] = defaultdict(dict)
        self.accumulated_update_history: Dict[Peer, Dict[int, nn.Module]] = defaultdict(dict)
        self.settings = settings
        self.data = Model.get_model_class(settings.model)().get_dataset_class()().all_test_data(120)
        self.aggregator: Aggregator = Aggregator.get_aggregator_class(settings.aggregator)()

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
        if p in self.models[host]:
            difference = model_difference(self.models[host][p], model)
        else:
            difference = model
        self.models[host][p] = model

        # Store model updates properly
        if p in self.accumulated_update_history[host]:
            self.accumulated_update_history[host][p] = model_sum(self.accumulated_update_history[host][p], difference)
        else:
            self.accumulated_update_history[host][p] = difference

        # Perform aggregation if all models are received
        if self.expecting_models == 0:
            self.expecting_models = self.settings.total_hosts * self.settings.peers_per_host
            peers = list(self.models.keys())
            models = reduce(add, map(lambda x: list(self.models[x].values()), peers))
            history = reduce(add, map(lambda x: list(self.accumulated_update_history[x].values()), peers))

            result = self.aggregator.aggregate(models, history)
            self.models.clear()

            result.eval()
            test_loss = 0
            test_corr = 0
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device_name)
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
