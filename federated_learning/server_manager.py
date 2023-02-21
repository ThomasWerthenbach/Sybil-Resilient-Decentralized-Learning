import json
from typing import Dict, List, Callable

from ipv8.types import Peer
from torch import nn
from torch.utils.data import DataLoader

from experiment_settings.settings import Settings
from federated_learning.manager import Manager
from ml.aggregators.aggregator import Aggregator
from ml.util import model_sum, deserialize_model, serialize_model


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
        self.deltas: Dict[Peer, nn.Module] = dict()
        self.accumulated_update_history: Dict[Peer, nn.Module] = dict()
        self.settings = settings
        self.aggregator: Aggregator = Aggregator.get_aggregator_class(settings.aggregator)()

    def get_all_models_and_reset(self) -> List[bytes]:
        result = list(map(lambda x: x[1], self.deltas.items()))
        self.deltas = dict()
        return result

    def receive_model(self, peer: Peer, info: bytes, delta: bytes):
        self.logger.info("Server received model")
        if peer in self.deltas:
            return

        # Store model properly
        self.deltas[peer] = deserialize_model(delta, self.settings)
        if peer in self.accumulated_update_history:
            self.accumulated_update_history[peer] = model_sum(self.accumulated_update_history[peer], self.deltas[peer])
        else:
            self.accumulated_update_history[peer] = self.deltas[peer]

        # Perform aggregation if all models are received
        if len(self.deltas) >= self.settings.total_peers:
            peers = list(self.deltas.keys())
            deltas = list(map(lambda x: self.deltas[x], peers))
            history = list(map(lambda x: self.accumulated_update_history[x], peers))

            result = self.aggregator.aggregate(deltas, history)

            # Reset deltas
            self.deltas = dict()

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
