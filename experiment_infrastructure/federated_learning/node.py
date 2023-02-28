from typing import List

from torch import nn
from torch.utils.data import DataLoader

from base_node import BaseNode
from experiment_infrastructure.experiment_settings.settings import Settings
from ml.models.model import Model


class Node(BaseNode):
    def __init__(self, model: Model, data: DataLoader, settings: Settings, _id: int):
        super().__init__()
        self.model = model
        self.data = data
        self.settings = settings
        self.id = _id

    def get_models(self) -> List[Model]:
        return [self.model]

    def start_next_epoch(self) -> None:
        self.train(self.model, self.data, self.settings)

    def get_ids(self) -> List[int]:
        return [self.id]

    def set_model(self, model: nn.Module):
        self.model = model
