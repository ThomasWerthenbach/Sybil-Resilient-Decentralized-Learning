import logging
from abc import abstractmethod

from ipv8.types import Peer

from experiments.experiment_settings.settings import Settings


class Manager:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def receive_model(self, peer_pk: Peer, info: bytes, delta: bytes):
        pass

    @abstractmethod
    def start_next_epoch(self):
        ...
