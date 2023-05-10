import json
import os
import random
from collections import defaultdict
from typing import Callable, List, Dict, Set

import pandas as pd
import torch.nn.functional as F

import torch.cuda
from ipv8.types import Peer

from experiment_infrastructure.attacks.attack import Attack
from experiment_infrastructure.decentralized_learning.base_node import BaseNode
from experiment_infrastructure.decentralized_learning.manager import Manager
from experiment_infrastructure.decentralized_learning.node import Node
from experiment_infrastructure.decentralized_learning.sybil import Sybil
from experiment_infrastructure.experiment_settings.settings import Settings
from ml.aggregators.aggregator import Aggregator
from ml.models.model import Model
from ml.util import serialize_model, deserialize_model


class NodeManager(Manager):
    TRAIN_MSG = 1
    GOSSIP_MSG = 2

    def __init__(self, settings: Settings, peer_id: int, me: Peer,
                 send_model: Callable[[Peer, int, bytes, bytes], None],
                 statistic_logger: Callable[[str, float], None]):
        super().__init__()
        self.peer_id = peer_id
        self.me = me
        self.round = 0
        self.settings = settings
        self.done_training = False
        self.send_model = send_model
        self.statistic_logger = statistic_logger
        self.test_data = Model.get_model_class(settings.model)().get_dataset_class()().all_test_data(120)
        self.attack_data = None
        if settings.sybil_attack:
            attack = Attack.get_attack_class(settings.sybil_attack_type)(settings)
            self.attack_rate_data = attack.transform_eval_data(self.test_data)
        else:
            self.attack_rate_data = None
        self.rounds: Dict[int, Dict[Peer, List[int]]] = defaultdict(lambda: defaultdict(list))

        self.edges: Dict[int, List[int]] = defaultdict(list)

        self.nodes: Dict[int, BaseNode] = dict()

        self.receiving_from: Set[int] = set()

        import torch
        self.logger.info(f"Cuda enabled: {torch.cuda.is_available()}")

        # Usage: self.adjacency_matrix[receiver][sender]
        adjacency_matrix: List[List[int]] = pd.read_csv(
            os.path.join(os.path.dirname(__file__), f"networks/{settings.network_layout}.csv")).values.tolist()
        for i in range(0, settings.peers_per_host):
            model = Model.get_model_class(settings.model)()
            node_id = self.get_node_id(i)

            if settings.sybil_attack and peer_id == settings.total_hosts and i == settings.peers_per_host - 1:
                # Sybil node
                dataset = model.get_dataset_class()() \
                    .get_peer_dataset(node_id,
                                      settings.total_hosts * settings.peers_per_host,
                                      settings.non_iid,
                                      alpha=settings.alpha,
                                      sybil_data_transformer=Attack.get_attack_class(settings.sybil_attack_type)(settings))
                self.nodes[node_id] = Sybil(model, dataset, settings, node_id)

                for sybil_id in self.nodes[node_id].get_ids():
                    all_peers = adjacency_matrix[sybil_id]
                    for j in [i for i in range(len(all_peers)) if all_peers[i] == 1]:
                        self.receiving_from.add(j)
                        self.edges[sybil_id].append(j)
            else:
                # Honest node
                dataset = model.get_dataset_class()().get_peer_dataset(node_id,
                                                                       settings.total_hosts * settings.peers_per_host,
                                                                       settings.non_iid,
                                                                       alpha=settings.alpha)
                self.nodes[node_id] = Node(model, dataset, settings, node_id)

                all_peers = adjacency_matrix[node_id]
                for j in [i for i in range(len(all_peers)) if all_peers[i] == 1]:
                    self.receiving_from.add(j)
                    self.edges[node_id].append(j)

        self.expecting_models = len(self.receiving_from)

    def start_next_epoch(self) -> None:
        for i in range(self.settings.peers_per_host):
            node_id = self.get_node_id(i)
            node = self.nodes[node_id]
            node.start_next_epoch(self.round)

            for model, _id in zip(node.get_models(), node.get_ids()):
                sent_to = set()
                for j in self.edges[_id]:
                    # We need to translate the peer id to a host id
                    host_id = min((j // self.settings.peers_per_host) + 1, self.settings.total_hosts)
                    # Host id's are 1-based
                    # Sybils will live on the host with the highest ID
                    if host_id not in sent_to:
                        sent_to.add(host_id)
                        self.send_model(self.me, host_id,
                                        json.dumps({'round': self.round, 'peer': _id, 'type': self.TRAIN_MSG}).encode(),
                                        serialize_model(model))

            if Aggregator.get_aggregator_class(self.settings.aggregator)().requires_gossip():
                neighbours = self.edges[node_id]
                for neighbour_id in neighbours:
                    history = node.get_random_neighbour_history(neighbour_id)
                    # Use round-robbin, not random
                    if history:
                        p, r, d, h = history
                        host_id = min((neighbour_id // self.settings.peers_per_host) + 1, self.settings.total_hosts)
                        self.send_model(self.me, host_id, json.dumps(
                            {'round': int(r), 'distant_peer': int(p), 'distance': int(d), 'type': self.GOSSIP_MSG,
                             'peer': int(node_id),
                             'destination': int(neighbour_id)}).encode(),
                                        serialize_model(h))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.done_training = True
        self.start_next_epoch_if_possible()

    def start_next_epoch_if_possible(self):
        if self.expecting_models == sum(list(map(lambda p: len(self.rounds[self.round][p]),
                                                 self.rounds[self.round].keys()))) and self.done_training:
            self.done_training = False
            for i in range(0, self.settings.peers_per_host):
                node_id = self.get_node_id(i)
                self.nodes[node_id].aggregate(self.round)
                accuracy, attack_rate = self.nodes[node_id].evaluate(self.test_data, self.attack_rate_data)
                if accuracy >= 0 and attack_rate >= 0:
                    self.statistic_logger(f"accuracy_{i}", accuracy)
                    self.statistic_logger(f"attack_rate_{i}", attack_rate)

            self.round += 1
            self.start_next_epoch()

    def receive_model(self, host: Peer, info: bytes, model: bytes):
        info = json.loads(info.decode())
        r = info['round']
        p = info['peer']
        t = info['type']

        if p not in self.receiving_from:
            self.logger.info("Received msg from peer that is not supposed to send")
            return

        if t == self.TRAIN_MSG:
            if len(self.rounds[r]) > self.expecting_models:
                self.logger.info(f"Received train update from {p} while not expecting any")
                return

            if self.round > r:
                self.logger.info(f"Received train update from previous round from {p}")
                return

            if p in self.rounds[r][host]:
                self.logger.info(f"Received train update from peer twice in same round from {p}")
                return
            self.rounds[r][host].append(p)

            model = deserialize_model(model, self.settings)

            for i in range(0, self.settings.peers_per_host):
                node_id = self.get_node_id(i)
                for j in self.nodes[node_id].get_ids():
                    if p in self.edges[j]:
                        self.logger.info(f"Received model from {p} for {node_id} (round {r})")
                        self.nodes[node_id].receive_model(model, p, r)

            self.start_next_epoch_if_possible()
        elif t == self.GOSSIP_MSG:
            destination = info['destination']
            distance = info['distance'] + 1  # We are one hop further away from the original sender
            distant_peer = info['distant_peer']
            for i in range(0, self.settings.peers_per_host):
                node_id = self.get_node_id(i)
                for j in self.nodes[node_id].get_ids():
                    if j == destination:
                        self.logger.info(f"Received history gossip from {p} for {destination} (from round {r})")
                        self.nodes[node_id].receive_distant_model(deserialize_model(model, self.settings), p, r, distance, distant_peer)
                        break

    def get_node_id(self, index: int) -> int:
        return self.settings.peers_per_host * (self.peer_id - 1) + index

    # # For debugging:
    # def get_att(self, model) -> float:
    #     with torch.no_grad():
    #         device_name = "cuda" if torch.cuda.is_available() else "cpu"
    #         device = torch.device(device_name)
    #         test_loss = 0
    #         test_corr = 0
    #         model.eval()
    #         model.to(device)
    #         for data, target in self.attack_rate_data:
    #             data, target = data.to(device), target.to(device)
    #             output = model(data)
    #             test_loss += F.nll_loss(output, target, reduction='sum').item()
    #             pred = output.argmax(dim=1, keepdim=True)
    #             test_corr += pred.eq(target.view_as(pred)).sum().item()
    #         test_loss /= len(self.attack_rate_data)
    #         test_att = 100. * test_corr / (len(self.attack_rate_data) * 120)
    #         return test_att
    #
    # def get_acc(self, model) -> float:
    #     with torch.no_grad():
    #         model.eval()
    #         test_loss = 0
    #         test_corr = 0
    #         device_name = "cuda" if torch.cuda.is_available() else "cpu"
    #         device = torch.device(device_name)
    #         model.to(device)
    #         for data, target in self.test_data:
    #             data, target = data.to(device), target.to(device)
    #             output = model(data)
    #             test_loss += F.nll_loss(output, target, reduction='sum').item()
    #             pred = output.argmax(dim=1, keepdim=True)
    #             test_corr += pred.eq(target.view_as(pred)).sum().item()
    #         test_loss /= len(self.test_data)
    #         test_acc = 100. * test_corr / (len(self.test_data) * 120)
    #         return test_acc
