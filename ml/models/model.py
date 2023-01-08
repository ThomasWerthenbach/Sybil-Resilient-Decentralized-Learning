from enum import Enum
from typing import Type

from torch import nn


class ModelType(Enum):
    """
    Defines the implemented machine learning models
    """
    MNIST = 'mnist'
    FashionMNIST = 'fashionmnist'
    EMNIST = 'emnist'


class Model(nn.Module):
    """
    Abstract class for a model
    """

    @staticmethod
    def get_model_class(model_name: ModelType) -> Type['Model']:
        """
        Returns the class of the model with the given name
        """
        if model_name == ModelType.MNIST:
            from ml.models.MNIST import MNIST
            return MNIST
        elif model_name == ModelType.FashionMNIST:
            from ml.models.FashionMNIST import FashionMNISTCNN
            return FashionMNISTCNN
        elif model_name == ModelType.EMNIST:
            from ml.models.EMNIST import EMNIST
            return EMNIST
        else:
            raise RuntimeError("Unknown model %s" % model_name)

    def freeze_non_output_layers(self):
        """
        Freezes all layers except the output layer
        """
        raise NotImplementedError()

    def replace_output_layer(self, new_output_layer: nn.Module):
        """
        Replaces the output layer of the model with the given one
        """
        raise NotImplementedError()

    def get_output_layer_weights(self) -> nn.Parameter:
        """
        Returns the weights of the output layer
        """
        raise NotImplementedError()
