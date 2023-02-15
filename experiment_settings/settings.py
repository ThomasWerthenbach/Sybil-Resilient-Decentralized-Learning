from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Settings:
    # ML job settings
    non_iid = True
    learning_rate = 0.001
    model = 'MNIST'
    aggregator = 'average'

    # Experiment settings
    total_peers: int
