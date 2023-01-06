from typing import Dict

from ipv8.community import Community
from ipv8.lazy_community import lazy_wrapper

from community.introduction_msg import IntroductionMsg
from community.ml_peer import MLPeer
from community.peer_manager import PeerManager
from community.settings import Settings
from community.weights_msg import WeightsMsg


class MLCommunity(Community):
    """
    Main community through which all messages are sent.
    """
    community_id = b"\x08\x07\xee\xa6\x97OgZ\xa1\x19\x14o\xfc\x87\xbe\x16/'Ka"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_message_handler(WeightsMsg, self.on_weights)
        self.add_message_handler(IntroductionMsg, self.on_introduction)
        self.peer_manager = PeerManager()
        self.me = MLPeer()

    def setup(self, settings: Settings):
        self.settings = settings

    def started(self):
        self.me.start_lifecycle(self)

    async def send_introductions(self, weights: Dict[any, float], data: any):
        """
        Only used by Repple.
        """

    async def on_introduction(self, peer, payload: IntroductionMsg):
        """
        Only used by Repple
        """

    async def send_weights_to_peers(self, weights: Dict[any, float]):
        # todo: send weights to peers stored in peermanager
        for peer in self.get_peers():
            self.ez_send(peer, WeightsMsg(weights=list(map(lambda x: WeightsMsg.Weight(x[0], x[1]), weights.items()))))

    @lazy_wrapper(WeightsMsg)
    def on_weights(self, peer, payload: WeightsMsg):
        # todo store weights in peer manager
        print("received weights from", peer, payload.weights)
