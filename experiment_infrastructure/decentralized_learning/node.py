import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from experiment_infrastructure.decentralized_learning.base_node import BaseNode
from experiment_infrastructure.experiment_settings.settings import Settings
from ml.aggregators.aggregator import Aggregator
from ml.models.model import Model
from ml.util import model_difference, model_sum


class HistoricalRecord:
    def __init__(self, round: int, distance: int, received_from: int, accumulated_update_history: nn.Module):
        self.round = round
        self.distance = distance
        self.accumulated_update_history = accumulated_update_history
        self.received_from = received_from


class Node(BaseNode):
    def __init__(self, model: Model, dataset: DataLoader, settings: Settings, node_id: int):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.settings = settings
        self.node_id = node_id

        self.models: Dict[int, Dict[int, nn.Module]] = defaultdict(dict)
        self.previous_models: Dict[int, nn.Module] = dict()

        self.history: Dict[int, HistoricalRecord] = dict()

        self.aggregator = Aggregator.get_aggregator_class(settings.aggregator)()

    def get_random_neighbour_history(self, for_peer: int) -> Tuple[int, int, int, nn.Module] | None:
        """
        Return a random neighbour's history. Note that nearby neighbours are preferred over distant ones.
        We utilize the exponential distribution y = lambda * e^(-lambda * x) to obtain the desired distribution.
        :return: A tuple of the peer id, the round number, the distance to the peer, and the model history.
        """
        if len(self.history) == 0:
            return None
        else:
            lamb = 1/1.25
            distribution = dict()
            for peer_id, history in self.history.items():
                distribution[peer_id] = lamb * np.exp(-lamb * (history.distance - 1))

            # Remove history of self
            if self.node_id in distribution:
                distribution.pop(self.node_id)

            # Remove history of for_peer itself
            if for_peer in distribution:
                distribution.pop(for_peer)

            # Remove history that we received from for_peer (can be found in received_from)
            for peer_id, history in self.history.items():
                if history.received_from == for_peer:
                    if peer_id in distribution:
                        distribution.pop(peer_id)

            if not distribution:
                return None

            distribution = {k: v / sum(distribution.values()) for k, v in distribution.items()}

            neighbour_id = np.random.choice(list(distribution.keys()), p=list(distribution.values()))
            history = self.history[neighbour_id]

            return neighbour_id, history.round, history.distance, history.accumulated_update_history

    def receive_distant_model(self, model: nn.Module, peer: int, round: int, distance: int, distant_peer: int):
        self.logger.info(f"{self.node_id} received distant model of {distant_peer} from {peer}")
        if distant_peer in self.history:
            if round > self.history[distant_peer].round and self.history[distant_peer].distance > 1:
                self.history[distant_peer] = HistoricalRecord(round, distance, peer, model)
            else:
                # Interesting case. If one were to apply a defense mechanism against spoofing model updates,
                # this would be the place to do so.
                pass
        else:
            self.history[distant_peer] = HistoricalRecord(round, distance, peer, model)

    def receive_model(self, model: nn.Module, peer: int, round: int):
        # Store model properly
        self.models[round][peer] = model
        # Calculate difference with previous model
        if peer in self.previous_models:
            difference = model_difference(self.previous_models[peer], model)
        else:
            difference = model

        # Store model updates properly
        if peer in self.history:
            self.history[peer] = HistoricalRecord(round, 1, -1,
                                                  model_sum(self.history[peer].accumulated_update_history, difference))
        else:
            self.history[peer] = HistoricalRecord(round, 1, -1, difference)

    def start_next_epoch(self, round: int) -> None:
        self.train(self.model, self.dataset, self.settings)
        self.receive_model(self.model, self.node_id, round)

    def aggregate(self, round: int) -> None:
        peers = list(filter(lambda p: p != self.node_id, list(self.models[round].keys())))
        models = list(map(lambda p: self.models[round][p], peers))
        history = list(map(lambda p: self.history[p], peers))
        # We add the history of distant neighbours as well
        history += list(map(lambda x: x[1], list(filter(lambda x: x[0] not in peers and x[0] != self.node_id, self.history.items()))))

        history = list(map(lambda h: h.accumulated_update_history, history))
        
        self.model = self.aggregator.aggregate(self.models[round][self.node_id], self.history[self.node_id].accumulated_update_history, models, history)
        self.previous_models = self.models[round]
        del self.models[round]

    def evaluate(self, test_data: DataLoader, attack_rate_data: DataLoader) -> Tuple[float, float]:
        with torch.no_grad():
            self.model.eval()
            test_loss = 0
            test_corr = 0
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device_name)
            self.model.to(device)
            for data, target in test_data:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_corr += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_data)
            test_acc = 100. * test_corr / (len(test_data) * 120)
            self.logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, test_corr, len(test_data) * 120, test_acc))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            test_att = 0.0
            if self.settings.sybil_attack:
                assert attack_rate_data is not None
                test_loss = 0
                test_corr = 0
                for data, target in attack_rate_data:
                    data, target = data.to(device), target.to(device)
                    output = self.model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    test_corr += pred.eq(target.view_as(pred)).sum().item()
                test_loss /= len(attack_rate_data)
                test_att = 100. * test_corr / (len(attack_rate_data) * 120)
                self.logger.info('Attack rate test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    test_loss, test_corr, len(attack_rate_data) * 120, test_att))

            return test_acc, test_att

    def get_models(self) -> List[Model]:
        return [self.model]

    def get_ids(self) -> List[int]:
        return [self.node_id]
