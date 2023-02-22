import json
from collections import defaultdict
from typing import Dict, Callable, List

from ipv8.types import Peer
from torch import nn
from torch.utils.data import DataLoader

from experiment_settings.settings import Settings
from federated_learning.manager import Manager
from ml.aggregators.aggregator import Aggregator
from ml.util import model_sum, deserialize_model, serialize_model, model_difference


class ServerManager(Manager):
    """
    1. Check if all nodes have submitted a model
    2. Aggregate
    3. Send aggregated model to all nodes
    """

    def __init__(self, settings: Settings, send_model: Callable[[Peer, bytes, bytes], None]):
        super().__init__()
        self.send_model = send_model
        self.round = 0
        self.expecting_models = settings.total_peers
        self.rounds: Dict[int, List[Peer]] = defaultdict(list)
        self.models: Dict[Peer, nn.Module] = dict()
        self.accumulated_update_history: Dict[Peer, nn.Module] = dict()
        self.settings = settings
        self.aggregator: Aggregator = Aggregator.get_aggregator_class(settings.aggregator)()

    def receive_model(self, peer: Peer, info: bytes, serialized_model: bytes):
        r = json.loads(info.decode())['round']
        if self.round != r:
            self.logger.info("Server received model with incorrect round number")
            return
        self.logger.info("Server received model")

        if peer in self.rounds[r]:
            self.logger.info("Server received model from peer twice")
            return
        self.rounds[r].append(peer)

        self.expecting_models -= 1

        # Store model properly
        model = deserialize_model(serialized_model, self.settings)
        # if peer in self.models:
        #     difference = model_difference(self.models[peer], model)
        # else:
        #     difference = model
        self.models[peer] = model

        # Store model updates properly
        # if peer in self.accumulated_update_history:
        #     self.accumulated_update_history[peer] = model_sum(self.accumulated_update_history[peer], difference)
        # else:
        #     self.accumulated_update_history[peer] = difference

        # Perform aggregation if all models are received
        if self.expecting_models == 0:
            self.expecting_models = self.settings.total_peers
            peers = list(self.models.keys())
            models = list(map(lambda x: self.models[x], peers))
            history = list()
            # history = list(map(lambda x: self.accumulated_update_history[x], peers))

            result = self.aggregator.aggregate(models, history)
            self.models = dict()

            # Send aggregated delta to all nodes
            self.logger.info("Server finished aggregation")
            self.round += 1
            for peer in peers:
                self.send_model(peer, json.dumps({'round': self.round}).encode(), serialize_model(result))

    def start_next_epoch(self):
        pass

    def get_dataset(self) -> DataLoader:
        raise NotImplementedError("Not applicable to role Server")

    def get_settings(self) -> Settings:
        return self.settings
