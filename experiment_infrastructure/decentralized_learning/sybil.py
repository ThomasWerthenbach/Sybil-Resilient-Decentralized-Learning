from collections import defaultdict
from typing import List, Tuple, Dict

from torch import nn
from torch.utils.data import DataLoader

from experiment_infrastructure.decentralized_learning.base_node import BaseNode
from experiment_infrastructure.experiment_settings.settings import Settings
from ml.aggregators.average import Average
from ml.models.model import Model


class Sybil(BaseNode):
    """
    Represents a malicious node launching a Sybil attack in the decentralized learning setting.
    Note that all created Sybils are also part of this node.
    """

    def receive_distant_model(self, model: nn.Module, peer: int, round: int, distance: int, for_peer: int) -> None:
        pass

    def get_random_neighbour_history(self, for_peer: int) -> Tuple[int, int, int, nn.Module] | None:
        return None

    def __init__(self, model: Model, data: DataLoader, settings: Settings, node_id: int):
        super().__init__()
        self.model = model
        self.data = data
        self.node_id = node_id
        self.settings = settings
        self.models: Dict[int, Dict[int, nn.Module]] = defaultdict(dict)

    def start_next_epoch(self, round: int) -> None:
        self.train(self.model, self.data, self.settings)
        for m, i in zip(self.get_models(), self.get_ids()):
            self.receive_model(m, i, round)

    def aggregate(self, round: int) -> None:
        """
        Malicious nodes aggregate simply through averaging regardless of the experiment configuration.
        """
        peers = list(self.models[round].keys())
        models = list(map(lambda p: self.models[round][p], peers))
        self.model = Average().aggregate(models, [])
        del self.models[round]

    def evaluate(self, test_data: DataLoader, attack_rate: DataLoader) -> Tuple[float, float]:
        return -1, -1

    def get_ids(self) -> List[int]:
        ids = [self.node_id]
        for i in range(self.settings.sybil_amount):
            ids.append(self.node_id + i + 1)
        return ids

    def get_models(self) -> List[Model]:
        return [self.model] * (self.settings.sybil_amount + 1)

    def receive_model(self, serialized_model: Model, peer: int, round: int) -> None:
        self.models[round][peer] = serialized_model
