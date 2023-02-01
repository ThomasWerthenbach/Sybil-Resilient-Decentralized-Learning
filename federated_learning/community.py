from asyncio import ensure_future
from enum import Enum

from ipv8.community import Community
from ipv8.types import Peer

from communication.util.eva.protocol import EVAProtocol
from communication.util.eva.result import TransferResult
from experiment_settings.settings import Settings
from federated_learning.manager import Manager
from federated_learning.node_manager import NodeManager
from federated_learning.server_manager import ServerManager
from federated_learning.sybil_manager import SybilManager


class Role(Enum):
    SERVER = 0
    NODE = 1
    SYBIL = 2


class FLCommunity(Community):
    """
    Main federated learning community.
    """
    community_id = b"\x08\x07\xee\xa6\x97OgZ\xa1\x19\x14o\xfc\x87\xbe\x16/'Ka"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eva = EVAProtocol(self, self.on_receive_model)
        self.manager: Manager | None = None

    def started(self):
        pass

    def start(self, settings: Settings, peer_id: int, role: Role, server: Peer):
        if role == Role.SERVER:
            # I am the server, my task is to wait for all nodes to finish training and then aggregate the model
            self.manager = ServerManager(settings, peer_id, self.send_model)
        elif role == Role.NODE:
            # I am a node, my task is to train on our own data and send our model to the server. Then we wait for the
            # aggregated model and train this again.
            self.manager = NodeManager(settings, peer_id, self.send_model, server)
        else:
            # I am the adversary, my task is to poison the aggregated model
            self.manager = SybilManager(settings, peer_id, self.send_model)
        self.register_task("start_lifecycle_" + str(peer_id), self.start_lifecycle, delay=0)

    async def on_receive_model(self, result: TransferResult):
        self.manager.receive_model(result.peer, result.data)

    async def start_lifecycle(self):
        self.manager.start_next_epoch()

    def send_model(self, peer: Peer, info: bytes, model: bytes):
        ensure_future(self.eva.send_binary(peer, info, model))
