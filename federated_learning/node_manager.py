import copy
import json
from typing import Callable

from ipv8.types import Peer
from torch.utils.data import DataLoader

from experiment_settings.settings import Settings
from federated_learning.manager import Manager
from ml.models.model import Model
from ml.util import serialize_model, deserialize_model


class NodeManager(Manager):
    """
    1. Train model on my own data
    2. Send model to server
    3. Wait for server to send the aggregated model
    4. Repeat
    """

    def __init__(self, settings: Settings, peer_id: int, me: Peer, send_model: Callable[[Peer, bytes, bytes], None],
                 server: Peer, statistic_logger: Callable[[str, float], None]):
        super().__init__()
        self.peer_id = peer_id
        self.me = me
        self.round = 0
        self.settings = settings
        self.send_model = send_model
        self.model = Model.get_model_class(settings.model)()
        self.server = server
        self.data = self.model.get_dataset_class()().get_peer_dataset(peer_id - 2, settings.total_peers, settings.non_iid)  # peer_id - 2, as peer_id's are 1-based and the server has id 1
        self.statistic_logger = statistic_logger
        # Used for producing results
        self.full_test_data = self.model.get_dataset_class()().all_test_data(120)

    def start_next_epoch(self) -> None:
        trained_model = copy.deepcopy(self.model)
        self.train(trained_model)
        self.send_model(self.server, json.dumps({'round': self.round}).encode(), serialize_model(trained_model))

    def receive_model(self, peer_pk: Peer, info: bytes, model: bytes):
        # 4.
        r = json.loads(info.decode())['round']
        if self.round >= r:
            # We already received a model during this round.
            return
        self.logger.info(f"Peer {self.me} received model from server {peer_pk} with hash {hash(model)}")
        self.round = r
        self.model = deserialize_model(model, self.settings)
        self.logger.info(f"Model requires grad: {self.model.requires_grad}")
        self.start_next_epoch()

    def get_dataset(self) -> DataLoader:
        return self.data

    def get_settings(self) -> Settings:
        return self.settings

    def get_all_test_data(self) -> DataLoader:
        return self.full_test_data

    def get_statistic_logger(self) -> Callable[[str, float], None]:
        return self.statistic_logger
