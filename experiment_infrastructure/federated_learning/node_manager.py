import copy
import json
from typing import Callable, List

import torch.cuda
from ipv8.types import Peer

from experiment_infrastructure.attacks.attack import Attack
from experiment_infrastructure.experiment_settings.settings import Settings
from experiment_infrastructure.federated_learning.base_node import BaseNode
from experiment_infrastructure.federated_learning.manager import Manager
from experiment_infrastructure.federated_learning.node import Node
from experiment_infrastructure.federated_learning.sybil import Sybil
from ml.models.model import Model
from ml.util import serialize_model, deserialize_model


class NodeManager(Manager):

    def __init__(self, settings: Settings, peer_id: int, send_model: Callable[[bytes, bytes], None]):
        super().__init__()
        self.round = 0
        self.settings = settings
        self.send_model = send_model
        self.nodes: List[BaseNode] = list()
        for i in range(settings.peers_per_host):
            model = Model.get_model_class(settings.model)()
            if settings.sybil_attack and peer_id == settings.total_hosts and i == settings.peers_per_host - 1:
                # Sybil node
                data = model.get_dataset_class()() \
                    .get_peer_dataset(settings.peers_per_host * (peer_id - 2) + i,
                                      # peer_id - 2, as peer_id's are 1-based and the server has id 1
                                      settings.total_hosts * settings.peers_per_host,
                                      settings.non_iid,
                                      sybil_data_transformer=Attack.get_attack_class(settings.sybil_attack_type)(settings))
                self.nodes.append(Sybil(model, data, settings, i))
            else:
                # Honest node
                data = model.get_dataset_class()() \
                    .get_peer_dataset(settings.peers_per_host * (peer_id - 2) + i,
                                      # peer_id - 2, as peer_id's are 1-based and the server has id 1
                                      settings.total_hosts * settings.peers_per_host,
                                      settings.non_iid)
                self.nodes.append(Node(model, data, settings, i))

    def start_next_epoch(self) -> None:
        for node in self.nodes:
            node.start_next_epoch()
            for model, _id in zip(node.get_models(), node.get_ids()):
                self.send_model(json.dumps({'round': self.round, 'peer': _id}).encode(), serialize_model(model))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def receive_model(self, peer_pk: Peer, info: bytes, model: bytes):
        r = json.loads(info.decode())['round']
        if self.round >= r:
            # We already received a model during this round.
            return
        self.round = r
        model = deserialize_model(model, self.settings)
        for node in self.nodes:
            node.set_model(copy.deepcopy(model))
        self.start_next_epoch()
