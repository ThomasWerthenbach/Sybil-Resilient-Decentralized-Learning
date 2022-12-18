from typing import List, Tuple

from ipv8.peer import Peer


class PeerManager:
    def __init__(self):
        self.peer_ids: List[Tuple[Peer, int]] = list()

    def add_peer(self, peer: Peer, peer_id: int):
        self.peer_ids.append((peer, peer_id))

    def get_peer_by_id(self, peer_id: int) -> Peer:
        for peer, _id in self.peer_ids:
            if _id == peer_id:
                return peer
        raise f"peer not found for id {peer_id}"

    def get_peer_id(self, peer: Peer) -> int:
        for p, _id in self.peer_ids:
            if p == peer:
                return _id
        raise "Peer not found"

    def get_peers(self) -> List[Peer]:
        return [p for p, _ in self.peer_ids]
