from typing import Callable

import torch
from ipv8.types import Peer
from torch.utils.data import DataLoader

from experiment_settings.settings import Settings
from federated_learning.manager import Manager
from federated_learning.util import deserialize_model, serialize_model
from ml.models.model import Model


class NodeManager(Manager):
    """
    1. Train model on my own data
    2. Send model to server
    3. Wait for server to send the aggregated model
    4. Repeat
    """

    def __init__(self, settings: Settings, peer_id: int, send_model: Callable[[Peer, bytes, bytes], None]):
        self.peer_id = peer_id
        self.settings = settings
        self.send_model = send_model
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        self.model = settings.model().to(device)
        self.data = self.model.get_dataset_class()().get_peer_dataset(peer_id, settings.total_peers, settings.non_iid)

    def start_next_epoch(self) -> None:
        self.train()
        # todo get the server
        server = None
        self.send_model(server, b'', serialize_model(self.model))

    def receive_model(self, peer_pk: Peer, model: bytes):
        # 4.
        self.model = deserialize_model(model, self.settings)
        self.start_next_epoch()

    def get_dataset(self) -> DataLoader:
        return self.data

    def get_model(self) -> Model:
        return self.model

    def get_settings(self) -> Settings:
        return self.settings
