from typing import Dict, List

from ipv8.types import Peer
from torch import nn
from torch.utils.data import DataLoader

from experiment_settings.settings import Settings
from federated_learning.manager import Manager
from federated_learning.util import deserialize_model
from ml.models.model import Model


class ServerManager(Manager):
    """
    1. Check if all nodes have submitted a model
    2. Aggregate
    3. Send aggregated model to all nodes
    """

    def __init__(self, settings: Settings, peer_id: int):
        self.peer_id = peer_id
        self.models: Dict[Peer, nn.Module] = dict()
        self.settings = settings

    def get_all_models_and_reset(self) -> List[bytes]:
        result = list(map(lambda x: x[1], self.models.items()))
        self.models = dict()
        return result

    def receive_model(self, peer_pk: Peer, model: bytes):
        self.models[peer_pk] = deserialize_model(model, self.settings)
        if len(self.models) > self.settings.total_peers:
            fools_gold()

    def start_next_epoch(self):
        pass

    def get_dataset(self) -> DataLoader:
        raise NotImplementedError("Not applicable to role Server")

    def get_model(self) -> Model:
        raise NotImplementedError("Not applicable to role Server")

    def get_settings(self) -> Settings:
        return self.settings
