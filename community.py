import random
from typing import List

from ipv8.community import Community
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_dataclass import dataclass

from base_settings import Settings


@dataclass(msg_id=1)
class RandomWalkMsg:
    path: List[int]
    hops: int
    backward: bool


class ReppleCommunity(Community):
    community_id = b"\x08\x07\xee\xa6\x97OgZ\xa1\x19\x14o\xfc\x87\xbe\x16/'Ka"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repple_id = -1
        self.settings = None

    def setup(self, repple_id: int, settings: Settings):
        self.repple_id = repple_id
        self.settings = settings

    def start(self):
        self.register_task("random_walk", self.random_walk, interval=1.0)

    @lazy_wrapper(RandomWalkMsg)
    def on_message(self, peer, payload: RandomWalkMsg):
        if payload.backward:
            pass
        else:
            payload.hops += 1
            payload.path.append(self.repple_id)

            if payload.hops < self.settings.max_random_walk_length:
                peer = random.choice(self.get_peers())
                self.ez_send(peer, payload)
            else:
                payload.backward = True
                self.ez_send(peer, payload)