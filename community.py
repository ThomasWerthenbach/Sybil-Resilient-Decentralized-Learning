import random
import time
from typing import List, Tuple, Dict, Set

from ipv8.community import Community
from ipv8.lazy_community import lazy_wrapper
from ipv8.messaging.payload_dataclass import dataclass
from ipv8.peer import Peer

from base_settings import Settings
from managers.peer_manager import PeerManager


@dataclass(msg_id=1)
class IntroductionMessage:
    peer_id: int


@dataclass(msg_id=2)
class RandomWalkMsg:
    initiator: int
    random_walk_id: int
    path: List[int]
    hops: int
    backward: bool


class ReppleCommunity(Community):
    community_id = b"\x08\x07\xee\xa6\x97OgZ\xa1\x19\x14o\xfc\x87\xbe\x16/'Ka"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.repple_id: int = -1
        self.settings: Settings = None
        self.propagated_random_walks: Dict[Tuple[int, int, int], List[int]] = dict()
        self.peer_manager = PeerManager()

    def setup(self, repple_id: int, settings: Settings):
        self.repple_id = repple_id
        self.settings = settings

    def started(self):
        self.register_task("introduce", self.introduce)

    def start(self):
        # testing
        if self.repple_id == 1:
            self.propagate_random_walk(None, RandomWalkMsg(initiator=self.repple_id,
                                                           random_walk_id=1,
                                                           path=[],
                                                           hops=0,
                                                           backward=False))

    async def introduce(self):
        introduced: Set[Peer] = set()
        while True:
            time.sleep(5)
            for peer in self.get_peers():
                if peer not in introduced:
                    introduced.add(peer)
                    self.ez_send(peer, IntroductionMessage(peer_id=self.repple_id))

    @lazy_wrapper(IntroductionMessage)
    def on_introduction(self, peer, payload: IntroductionMessage):
        self.peer_manager.add_peer(peer, payload.peer_id)

    @lazy_wrapper(RandomWalkMsg)
    def on_message(self, peer, payload: RandomWalkMsg):
        if payload.backward:
            self.propagate_random_walk_backward(peer, payload)
        else:
            self.propagate_random_walk(peer, payload)

    def propagate_random_walk(self, peer, payload: RandomWalkMsg):
        payload.hops += 1
        payload.path.append(self.repple_id)

        if payload.hops < self.settings.max_random_walk_length:
            # todo only choose from evaluated peers
            peer = random.choice(self.peer_manager.get_peers())
            self.propagated_random_walks[(payload.initiator, payload.random_walk_id, payload.hops)] = payload.path
            self.ez_send(peer, payload)
        else:
            payload.backward = True
            self.ez_send(peer, payload)

    def propagate_random_walk_backward(self, peer, payload: RandomWalkMsg):
        payload.hops -= 1

        valid = self.verify_path(payload, peer)
        if not valid:
            return

        if payload.hops > 0:
            peer = self.peer_manager.get_peer_by_id(payload.path[payload.hops - 1])
            self.ez_send(peer, payload)
        else:
            # todo random walk complete!
            print(payload.path)

    def verify_path(self, payload: RandomWalkMsg, from_peer):
        propagated_path = self.propagated_random_walks[(payload.initiator, payload.random_walk_id, payload.hops)]
        # verify if payload.path starts with propagated_path
        if propagated_path[:len(payload.path)] != payload.path:
            return False
        # verify if from_peer is indeed the peer that sent us the message
        peer_id = self.peer_manager.get_peer_id(from_peer)
        if peer_id != payload.path[payload.hops - 1]:
            return False
        return True
