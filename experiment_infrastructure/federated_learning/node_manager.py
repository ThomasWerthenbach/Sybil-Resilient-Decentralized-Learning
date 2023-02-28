import copy
import json
from typing import Callable, List

import torch.cuda
from ipv8.types import Peer

from experiment_infrastructure.experiment_settings.settings import Settings
from experiment_infrastructure.federated_learning.manager import Manager
from experiment_infrastructure.federated_learning.node import Node
from ml.models.model import Model
from ml.util import serialize_model, deserialize_model


class NodeManager(Manager):

    def __init__(self, settings: Settings, peer_id: int, me: Peer, send_model: Callable[[bytes, bytes], None],
                 server: Peer, statistic_logger: Callable[[str, float], None]):
        super().__init__()
        self.me = me
        self.round = 0
        self.settings = settings
        self.send_model = send_model
        self.server = server
        self.nodes: List[Node] = list()
        for i in range(settings.peers_per_host):
            model = Model.get_model_class(settings.model)()
            data = model.get_dataset_class()() \
                .get_peer_dataset(settings.peers_per_host * (peer_id - 2) + i,
                                  # peer_id - 2, as peer_id's are 1-based and the server has id 1
                                  settings.total_hosts * settings.peers_per_host,
                                  settings.non_iid)
            self.nodes.append(Node(model, data, settings, i))
        self.statistic_logger = statistic_logger

    def start_next_epoch(self) -> None:
        for node in self.nodes:
            node.start_next_epoch()
            for model in node.get_models():
                self.send_model(json.dumps({'round': self.round, 'peer': node.get_id()}).encode(),
                                serialize_model(model))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def receive_model(self, peer_pk: Peer, info: bytes, model: bytes):
        r = json.loads(info.decode())['round']
        if self.round >= r:
            # We already received a model during this round.
            return
        self.logger.info(f"Node {self.me} received model from server {peer_pk} with hash {hash(model)}")
        self.round = r
        model = deserialize_model(model, self.settings)
        for node in self.nodes:
            node.set_model(copy.deepcopy(model))
        self.start_next_epoch()
