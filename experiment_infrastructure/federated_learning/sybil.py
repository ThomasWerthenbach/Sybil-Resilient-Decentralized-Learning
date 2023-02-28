from typing import List

from torch import nn
from torch.utils.data import DataLoader

from experiment_infrastructure.experiment_settings.settings import Settings
from experiment_infrastructure.federated_learning.base_node import BaseNode
from ml.models.model import Model


class Sybil(BaseNode):

    def __init__(self, model: Model, data: DataLoader, settings: Settings, node_id: int):
        super().__init__()
        self.model = model
        self.data = data
        self.node_id = node_id
        self.settings = settings

    def start_next_epoch(self) -> None:
        self.train(self.model, self.data, self.settings)

    def get_ids(self) -> List[int]:
        ids = [self.node_id]
        for i in range(self.settings.sybil_amount):
            ids.append(self.node_id + i + 1)
        return ids

    def get_models(self) -> List[Model]:
        return [self.model] * (self.settings.sybil_amount + 1)

    def set_model(self, model: nn.Module) -> None:
        self.model = model
