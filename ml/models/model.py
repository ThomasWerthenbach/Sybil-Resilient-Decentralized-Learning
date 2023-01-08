from abc import ABC, abstractmethod
from enum import Enum
from typing import Type, List

from torch import nn, Tensor


class ModelType(Enum):
    """
    Defines the implemented machine learning models
    """
    MNIST = 'mnist'
    FashionMNIST = 'fashionmnist'
    EMNIST = 'emnist'


class Model(nn.Module, ABC):
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

    @abstractmethod
    def prepare_model_for_transfer_learning(self, num_classes: int):
        """
        Freezes all layers except the output layer
        """

    @abstractmethod
    def replace_output_layer(self, new_output_layer: Tensor):
        """
        Replaces the output layer of the model with the given one
        """

    @abstractmethod
    def get_output_layer_weights(self) -> Tensor:
        """
        Returns the weights of the output layer
        """

    def get_serialized_layer_weights(self) -> List[List[float]]:
        output_layer: nn.Parameter = self.own_model.get_output_layer_weights()
        result = []
        for i in range(len(output_layer)):
            weights = []
            for j in range(len(output_layer[i])):
                weights.append(output_layer[i][j])
            result.append(weights)
        return result
