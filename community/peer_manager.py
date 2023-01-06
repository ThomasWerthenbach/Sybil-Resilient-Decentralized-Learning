from typing import List

from ipv8.peer import Peer


class PeerManager:
    def __init__(self):
        self.approached_peers: List[Peer] = []
