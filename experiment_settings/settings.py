from typing import Type

from ml.aggregators.aggregator import Aggregator
from ml.aggregators.average import Average
from ml.models.MNIST import MNIST
from ml.models.model import Model


class Settings:
    # Network settings
    max_rounds = 100
    total_peers = 2

    # ML job settings
    non_iid = False
    learning_rate = 0.001
    model: Type[Model] = MNIST
    aggregator: Type[Aggregator] = Average
