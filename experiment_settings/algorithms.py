from enum import Enum


class Algorithm(Enum):
    """
    Defines the supported federated machine learning algorithms
    """
    AVERAGE = "average"
    REPPLE = "repple"
    BRISTLE = "bristle"
