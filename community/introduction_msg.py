from typing import List

from ipv8.messaging.payload_dataclass import dataclass


@dataclass(msg_id=2)
class IntroductionMsg:
    """
    Message used to communicate weights used in random flood calculation.
    """
    # WIP

    @dataclass
    class Weight:
        p: str
        w: str  # floats are not supported in ipv8

    weights: List[Weight]  # peers and weights
    data: str  # todo sending data may not be supported by ipv8 -> use EVA protocol by Martijn?
