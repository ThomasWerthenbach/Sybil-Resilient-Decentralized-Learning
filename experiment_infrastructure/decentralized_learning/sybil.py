from typing import List, Tuple

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

    def __init__(self, model: Model, data: DataLoader, settings: Settings, node_id: int):
        super().__init__()
        self.model = model
        self.data = data
        self.node_id = node_id
        self.settings = settings
        self.models = list()

    def start_next_epoch(self) -> None:
        self.train(self.model, self.data, self.settings)

    def aggregate(self) -> None:
        """
        Malicious nodes aggregate simply through averaging regardless of the experiment configuration.
        """
        self.model = Average().aggregate(self.models + self.get_models(), [])

    def evaluate(self, test_data: DataLoader) -> Tuple[float, float]:
        return -1, -1

    def get_ids(self) -> List[int]:
        ids = [self.node_id]
        for i in range(self.settings.sybil_amount):
            ids.append(self.node_id + i + 1)
        return ids

    def get_models(self) -> List[Model]:
        return [self.model] * (self.settings.sybil_amount + 1)

    def receive_model(self, serialized_model: Model, peer: int) -> None:
        if peer < self.node_id:
            # We don't store Sybil models
            self.models.append(serialized_model)
