import pickle

import torch

from experiment_settings.settings import Settings


def serialize_model(model: torch.nn.Module) -> bytes:
    return pickle.dumps(model.state_dict())


def deserialize_model(serialized_model: bytes, settings: Settings) -> torch.nn.Module:
    model = settings.model()
    model.load_state_dict(pickle.loads(serialized_model))
    return model
