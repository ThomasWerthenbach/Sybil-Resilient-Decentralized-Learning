from typing import List

from ipv8.messaging.payload_dataclass import dataclass


@dataclass(msg_id=1)
class RecommendationsMsg:
    """
    Message used to communicate weights used in random flood calculation.
    """

    @dataclass
    class Recommendation:
        p: str
        w: str  # floats are not supported in ipv8

    weights: List[Recommendation]  # peers and weights