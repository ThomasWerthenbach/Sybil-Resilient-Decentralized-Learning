import json
import os
from collections import defaultdict
from typing import Callable, List, Dict, Set

import pandas as pd
import torch.cuda
from ipv8.types import Peer

from experiment_infrastructure.decentralized_learning.manager import Manager
from experiment_infrastructure.decentralized_learning.node import Node
from experiment_infrastructure.experiment_settings.settings import Settings
from ml.models.model import Model
from ml.util import serialize_model, deserialize_model


class NodeManager(Manager):

    def __init__(self, settings: Settings, peer_id: int, me: Peer, send_model: Callable[[int, bytes, bytes], None],
                 statistic_logger: Callable[[str, float], None]):
        super().__init__()
        self.peer_id = peer_id
        self.me = me
        self.round = 0
        self.settings = settings
        self.send_model = send_model
        self.statistic_logger = statistic_logger
        self.test_data = Model.get_model_class(settings.model)().get_dataset_class()().all_test_data(120)
        self.rounds: Dict[int, Dict[Peer, List[int]]] = defaultdict(lambda: defaultdict(list))

        self.edges: Dict[int, List[int]] = defaultdict(list)

        self.nodes = dict()

        self.receiving_from: Set[int] = set()

        # Usage: self.adjacency_matrix[receiver][sender]

        adjacency_matrix: List[List[int]] = pd.read_csv(os.path.join(os.path.dirname(__file__), "100.csv")).values.tolist()
        for i in range(0, settings.peers_per_host):
            node_id = self.get_node_id(i)

            model = Model.get_model_class(settings.model)()
            dataset = model.get_dataset_class()().get_peer_dataset(node_id,
                                                                   settings.total_hosts * settings.peers_per_host,
                                                                   settings.non_iid)
            self.nodes[node_id] = Node(model, dataset, settings)

            all_peers = adjacency_matrix[node_id]
            for j in [i for i in range(len(all_peers)) if all_peers[i] == 1]:
                self.receiving_from.add(j)
                self.edges[i].append(j)

        self.expecting_models = len(self.receiving_from)

    def start_next_epoch(self) -> None:
        for i in range(self.settings.peers_per_host):
            trained_model = self.nodes[self.get_node_id(i)].start_next_epoch()
            serialized_model = serialize_model(trained_model)
            for j in self.edges[i]:
                self.send_model(j, json.dumps({'round': self.round, 'peer': i}).encode(),
                                serialized_model)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def receive_model(self, host: Peer, info: bytes, model: bytes):
        info = json.loads(info.decode())
        r = info['round']
        p = info['peer']

        if p not in self.receiving_from:
            self.logger.info("Received model from peer that is not supposed to send")
            return

        if self.expecting_models <= 0:
            self.logger.info("Received model while not expecting any")
            return

        if self.round >= r:
            self.logger.info("Received model from previous round")
            return

        if p in self.rounds[r][host]:
            self.logger.info("Received model from peer twice in same round")
            return
        self.rounds[r][host].append(p)

        model = deserialize_model(model, self.settings)
        for i in range(0, self.settings.peers_per_host):
            node_id = self.get_node_id(i)
            if p in self.edges[node_id]:
                self.nodes[node_id].receive_model(model, p)

        self.expecting_models -= 1
        if self.expecting_models == 0:
            for i in range(0, self.settings.peers_per_host):
                self.nodes[i].aggregate()
                accuracy = self.nodes[i].evaluate(self.test_data)
                self.statistic_logger(f"accuracy_{i}", accuracy)

            self.expecting_models = len(self.receiving_from)
            self.round += 1
            self.start_next_epoch()

    def get_node_id(self, index: int) -> int:
        return self.settings.peers_per_host * (self.peer_id - 1) + index
