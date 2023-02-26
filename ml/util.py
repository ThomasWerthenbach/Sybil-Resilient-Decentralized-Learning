import copy
import pickle

import torch
from torch import nn

from experiment_infrastructure.experiment_settings.settings import Settings


def model_difference(prior_model: nn.Module, new_model: nn.Module) -> nn.Module:
    """
    Calculates the difference between two models
    """
    prior_model = prior_model.to('cpu')
    new_model = new_model.to('cpu')
    delta = copy.deepcopy(prior_model)
    with torch.no_grad():
        for p1, p2, p3 in zip(delta.parameters(), prior_model.parameters(), new_model.parameters()):
            p1.mul_(0)
            p1.add_(p3.data - p2.data)
    return delta


def model_sum(first_model: nn.Module, second_model: nn.Module) -> nn.Module:
    """
    Calculates the sum of two models
    """
    first_model = first_model.to('cpu')
    second_model = second_model.to('cpu')
    delta = copy.deepcopy(first_model)
    with torch.no_grad():
        for p1, p2, p3 in zip(delta.parameters(), first_model.parameters(), second_model.parameters()):
            p1.mul_(0)
            p1.add_(p3.data + p2.data)
    return delta


def serialize_model(model: torch.nn.Module) -> bytes:
    return pickle.dumps(model.state_dict())


def deserialize_model(serialized_model: bytes, settings: Settings) -> torch.nn.Module:
    from ml.models.model import Model
    model = Model.get_model_class(settings.model)()
    model.load_state_dict(pickle.loads(serialized_model))
    return model
