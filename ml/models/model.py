from typing import Type

from torch import nn


class Model(nn.Module):
    """
    Abstract class for a model
    """

    @staticmethod
    def get_model_class(model_name: str) -> Type['Model']:
        """
        Returns the class of the model with the given name
        """
        if model_name.lower() == 'mnist':
            from ml.models.MNIST import MNIST
            return MNIST
        elif model_name.lower() == 'fashionmnist':
            from ml.models.FashionMNIST import FashionMNISTCNN
            return FashionMNISTCNN
        else:
            raise RuntimeError("Unknown model %s" % model_name)
