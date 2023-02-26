import asyncio
import traceback
from asyncio import ensure_future, Future

from ipv8.community import Community

from communication.util.eva.protocol import EVAProtocol
from communication.util.eva.result import TransferResult
from experiment_infrastructure.experiment_settings.settings import Settings
from experiment_infrastructure.decentralized_learning.manager import Manager
from experiment_infrastructure.decentralized_learning.node_manager import NodeManager
from experiment_infrastructure.decentralized_learning.sybil_manager import SybilManager


class DLCommunity(Community):
    """
    Main decentralized learning community.
    """
    community_id = b"\x08\x07\xee\xa6\x97OgZ\xa1\x19\x14o\xfc\x87\xbe\x16/'Ka"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eva = EVAProtocol(self, self.on_receive_model)
        self.manager: Manager | None = None
        self.experiment_module = None
        self.settings: Settings | None = None

    def started(self):
        pass

    def assign_node(self, host_id: int, settings: Settings, experiment_module):
        self.settings = settings
        self.manager = NodeManager(settings, host_id, self.my_peer, self.send_model,
                                   experiment_module.autoplot_add_point)
        self.experiment_module = experiment_module
        for i in range(settings.peers_per_host):
            self.experiment_module.autoplot_create(f"accuracy_{i}")
        self.register_task("start_lifecycle_" + str(host_id), self.start_lifecycle, delay=0)

    def assign_sybil(self, peer_id: int, settings: Settings, experiment_module):
        # I am the adversary, my task is to poison the aggregated model
        self.manager = SybilManager(settings, peer_id, self.send_model)

    def log(self, message: str):
        self.logger.info(message)

    async def on_receive_model(self, result: TransferResult):
        try:
            self.manager.receive_model(result.peer, result.info, result.data)
        except:
            tb = traceback.format_exc()
            self.logger.error(f"Failed to receive model from {result.peer} Exception:\n {tb}")

    async def start_lifecycle(self):
        self.manager.start_next_epoch()

    def on_eva_send_done(self, future: Future, peer: int, serialized_response: bytes, binary_data: bytes, nonce: int):
        if future.cancelled():  # Do not reschedule if the future was cancelled
            return

        if future.exception():
            self.logger.warning(f"Transfer to {peer} failed, scheduling it again (Exception: %s)", future.exception())
            # The transfer failed - try it again after some delay
            ensure_future(asyncio.sleep(1)).add_done_callback(
                lambda _: self.send_model(peer, serialized_response, binary_data, nonce))

    def send_model(self, peer: int, info: bytes, model: bytes, nonce: int = None):
        try:
            if nonce is None:
                nonce = self.eva.get_nonce()

            if peer == self.experiment_module.my_id:
                self.register_task("send_to_self_" + str(nonce), self.on_receive_model,
                                   TransferResult(self.my_peer, info, model, nonce), delay=0)
            else:
                future = ensure_future(self.eva.send_binary(self.experiment_module.get_peer(str(peer)), info, model, nonce))
                future.add_done_callback(lambda f: self.on_eva_send_done(f, peer, info, model, nonce))
        except:
            tb = traceback.format_exc()
            self.logger.error(f"Failed to send model to {peer} Exception:\n {tb}")
