from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Settings:
    # ML job settings
    non_iid: bool
    learning_rate: float
    momentum: float
    model: str
    aggregator: str

    # Experiment settings
    total_peers: int
