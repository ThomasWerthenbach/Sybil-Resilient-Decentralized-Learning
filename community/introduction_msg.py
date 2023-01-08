from typing import List

from ipv8.messaging.payload_dataclass import dataclass


@dataclass(msg_id=2)
class IntroductionMsg:
    """
    Message used to communicate weights used in random flood calculation.
    """
    # todo
