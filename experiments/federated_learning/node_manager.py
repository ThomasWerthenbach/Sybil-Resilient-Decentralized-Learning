import copy
import json
from typing import Callable, List

import torch.cuda
from ipv8.types import Peer
from torch.utils.data import DataLoader

from experiments.experiment_settings.settings import Settings
from experiments.federated_learning.manager import Manager
from ml.models.model import Model
from ml.util import serialize_model, deserialize_model


class NodeManager(Manager):

    def __init__(self, settings: Settings, peer_id: int, me: Peer, send_model: Callable[[Peer, bytes, bytes], None],
                 server: Peer, statistic_logger: Callable[[str, float], None]):
        super().__init__()
        self.peer_id = peer_id
        self.me = me
        self.round = 0
        self.settings = settings
        self.send_model = send_model
        self.models: List[Model] = [Model.get_model_class(settings.model)() for _ in range(settings.peers_per_host)]
        self.server = server
        self.data: List[DataLoader] = list()
        for i, model in enumerate(self.models):
            self.data.append(model.get_dataset_class()() \
                             .get_peer_dataset(settings.peers_per_host * (peer_id - 2) + i,  # peer_id - 2, as peer_id's are 1-based and the server has id 1
                                               settings.total_hosts * settings.peers_per_host,
                                               settings.non_iid))
        self.statistic_logger = statistic_logger

    def start_next_epoch(self) -> None:
        for i, (model, data) in enumerate(zip(self.models, self.data)):
            self.train(model, data)
            self.send_model(self.server, json.dumps({'round': self.round, 'peer': i}).encode(), serialize_model(model))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def receive_model(self, peer_pk: Peer, info: bytes, model: bytes):
        r = json.loads(info.decode())['round']
        if self.round >= r:
            # We already received a model during this round.
            return
        self.logger.info(f"Peer {self.me} received model from server {peer_pk} with hash {hash(model)}")
        self.round = r
        model = deserialize_model(model, self.settings)
        for i in range(len(self.models)):
            self.models[i] = copy.deepcopy(model)
        self.start_next_epoch()

    def get_settings(self) -> Settings:
        return self.settings
