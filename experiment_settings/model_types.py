from enum import Enum


class ModelType(Enum):
    """
    Defines the implemented machine learning models
    """
    MNIST = 'mnist'
    FashionMNIST = 'fashionmnist'
    EMNIST = 'emnist'