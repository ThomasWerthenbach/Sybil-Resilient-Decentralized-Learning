import asyncio
from asyncio import ensure_future, Future

from ipv8.community import Community
from ipv8.types import Peer

from communication.util.eva.protocol import EVAProtocol
from communication.util.eva.result import TransferResult
from experiment_settings.settings import Settings
from federated_learning.manager import Manager
from federated_learning.node_manager import NodeManager
from federated_learning.server_manager import ServerManager
from federated_learning.sybil_manager import SybilManager


class FLCommunity(Community):
    """
    Main federated learning community.
    """
    community_id = b"\x08\x07\xee\xa6\x97OgZ\xa1\x19\x14o\xfc\x87\xbe\x16/'Ka"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #todo
        # EVA protocol can be improved:
        # - 1. Old acknowledgements need to be able to cancel re-transmissions.
        # - 2. EVA will use a sliding-window mechanism to limit the amount of data we potentially have to re-sent.
        self.eva = EVAProtocol(self, self.on_receive_model)
        self.manager: Manager | None = None

    def started(self):
        pass

    def assign_server(self, settings: Settings):
        # I am the server, my task is to wait for all nodes to finish training and then aggregate the model
        self.manager = ServerManager(settings, self.send_model)

    def assign_node(self, peer_id: int, server: Peer, settings: Settings):
        # I am a node, my task is to train on our own data and send our model to the server. Then we wait for the
        # aggregated model and train this again.
        self.manager = NodeManager(settings, peer_id, self.my_peer, self.send_model, server)
        self.register_task("start_lifecycle_" + str(peer_id), self.start_lifecycle, delay=0)

    def assign_sybil(self, peer_id: int, server: Peer, settings: Settings):
        # I am the adversary, my task is to poison the aggregated model
        self.manager = SybilManager(settings, peer_id, self.send_model)

    def log(self, message: str):
        self.logger.info(message)

    async def on_receive_model(self, result: TransferResult):
        self.manager.receive_model(result.peer, result.info, result.data)

    async def start_lifecycle(self):
        self.manager.start_next_epoch()

    def on_eva_send_done(self, future: Future, peer: Peer, serialized_response: bytes, binary_data: bytes, nonce: int):
        if future.cancelled():  # Do not reschedule if the future was cancelled
            return

        if future.exception():
            self.logger.warning(f"Transfer to {peer} failed, scheduling it again (Exception: %s)", future.exception())
            # The transfer failed - try it again after some delay
            ensure_future(asyncio.sleep(1)).add_done_callback(
                lambda _: self.send_model(peer, serialized_response, binary_data, nonce))

    def send_model(self, peer: Peer, info: bytes, model: bytes, nonce: int = None):
        if nonce is None:
            nonce = self.eva.get_nonce()
        future = ensure_future(self.eva.send_binary(peer, info, model, nonce))
        future.add_done_callback(lambda f: self.on_eva_send_done(f, peer, info, model, nonce))
