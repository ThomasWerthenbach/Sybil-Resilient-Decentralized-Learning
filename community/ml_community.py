from typing import Dict, List

from ipv8.community import Community
from ipv8.lazy_community import lazy_wrapper
from ipv8.peer import Peer

from community.introduction_msg import IntroductionMsg
from ml.ml_peer import MLPeer
from community.peer_manager import PeerManager
from community.settings import Settings
from community.recommendations_msg import RecommendationsMsg
from community.weights_msg import WeightsMsg


class MLCommunity(Community):
    """
    Main community through which all messages are sent.
    """
    community_id = b"\x08\x07\xee\xa6\x97OgZ\xa1\x19\x14o\xfc\x87\xbe\x16/'Ka"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_message_handler(RecommendationsMsg, self.on_recommendations)
        self.add_message_handler(IntroductionMsg, self.on_introduction)
        self.peer_manager = PeerManager()
        self.me = None

    def start(self, settings: Settings, peer_id: int):
        self.me = MLPeer(peer_id, settings)
        self.me.start_lifecycle(self)

    async def send_introductions(self, weights: Dict[any, float], data: any):
        """
        Only used by Repple.
        """

    @lazy_wrapper(IntroductionMsg)
    def on_introduction(self, peer, payload: IntroductionMsg):
        """
        Only used by Repple
        """

    async def send_model_to_peers(self, model: List[List[str]]):
        """
        Used to send models to peers
        :param model: Stringified weights of the output layer
        """
        for peer in self.get_peers():
            self.ez_send(peer, WeightsMsg(weights=list(map(WeightsMsg.ClassWeight, model))))

    async def send_recommendations(self, weights: Dict[str, str]):
        # todo: send weights to peers stored in peermanager
        for peer in self.get_peers():
            self.ez_send(peer, RecommendationsMsg(weights=list(map(lambda x: RecommendationsMsg.Recommendation(x[0], x[1]), weights.items()))))

    @lazy_wrapper(RecommendationsMsg)
    def on_recommendations(self, peer, payload: RecommendationsMsg):
        # todo store weights in peer manager
        print("received weights from", peer, payload.weights)

    @lazy_wrapper(WeightsMsg)
    def on_weights(self, peer, payload: WeightsMsg):
        # todo check if we have accepted the peers connection request IFF we are using Repple
        self.peer_manager.on_weights(peer, payload)

    def get_other_models(self) -> Dict[Peer, List[List[float]]]:
        return self.peer_manager.get_and_reset_weights()
