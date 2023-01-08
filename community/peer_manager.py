from typing import List, Dict

from ipv8.peer import Peer

from community.weights_msg import WeightsMsg


class PeerManager:
    def __init__(self):
        self.approached_peers: List[Peer] = list()
        self.latest_model_weights: Dict[Peer, List[List[float]]] = dict()

    def on_weights(self, peer, weights: WeightsMsg):
        self.latest_model_weights[peer] = list(map(lambda x: list(map(float, x.w)), weights.weights))

    def get_and_reset_weights(self) -> Dict[Peer, List[List[float]]]:
        result = self.latest_model_weights
        self.latest_model_weights = dict()
        return result
