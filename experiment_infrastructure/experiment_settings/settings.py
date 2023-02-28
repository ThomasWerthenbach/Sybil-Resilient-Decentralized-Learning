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
    epochs: int

    # Experiment settings
    total_hosts: int
    peers_per_host: int
    network_layout: str

    # Sybil settings
    sybil_amount: int
    sybil_attack: str
