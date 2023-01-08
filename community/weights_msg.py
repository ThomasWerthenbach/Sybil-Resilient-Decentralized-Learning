from typing import List

from ipv8.messaging.payload_dataclass import dataclass


@dataclass(msg_id=3)
class WeightsMsg:
    """
    Message used to communicate the new trained model weights.
    """
    @dataclass
    class ClassWeight:
        w: List[str]

    weights: List[ClassWeight]
