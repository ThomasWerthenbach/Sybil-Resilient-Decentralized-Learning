from experiment_settings.algorithms import Algorithm
from experiment_settings.model_types import ModelType


class Settings:
    max_rounds = 100
    total_peers = 2
    non_iid = False

    learning_rate = 0.001

    min_random_flood_value = 1e-4
    transitivity_decay = 0.01

    model: ModelType = ModelType.EMNIST
    pretrained_model_location = "../transfer_scripts/mnist-10-epoch.pth"

    algorithm: Algorithm = Algorithm.AVERAGE
