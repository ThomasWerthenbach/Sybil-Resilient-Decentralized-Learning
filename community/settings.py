from ml.executor.executor import Algorithm
from ml.models.model import ModelType


class Settings:
    max_rounds = 100

    min_random_flood_value = 1e-4
    transitivity_decay = 0.01

    model: ModelType = ModelType.EMNIST
    # pretrained_model_location = "models/emnist/emnist.pth"

    algorithm: Algorithm = Algorithm.AVERAGE
