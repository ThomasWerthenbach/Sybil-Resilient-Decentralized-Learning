from abc import ABC, abstractmethod
from enum import Enum
from typing import Type, List, Dict

from ipv8.peer import Peer

from community.settings import Settings
from ml.executor.average import AverageExecutor


class Algorithm(Enum):
    """
    Defines the supported federated machine learning algorithms
    """
    AVERAGE = "average"
    REPPLE = "repple"
    BRISTLE = "bristle"


class Executor(ABC):
    def __init__(self, settings: Settings):
        self._settings = settings

    @staticmethod
    def get_executor_class(executor_name: Algorithm) -> Type['Executor']:
        if executor_name == Algorithm.AVERAGE:
            return AverageExecutor
        else:
            raise ValueError("Unknown executor: " + executor_name.value)

    @abstractmethod
    def prepare_model(self):
        """
        Load pretrained model, replace output layer and freeze all non-output layers.
        """

    @abstractmethod
    def train(self):
        """
        Train the model for one epoch
        """

    @abstractmethod
    def get_model_weights(self) -> List[List[float]]:
        """
        Get the weights of the output layer of the trained model
        """

    @abstractmethod
    def prioritize_other_models(self, other_models: Dict[Peer, List[List[float]]]) -> List[List[List[float]]]:
        """
        Prioritize/filter other models based on algorithm-specific criteria
        """

    @abstractmethod
    def integrate_models(self, prioritized_models):
        """
        Integrate the prioritized models into the local model
        """
