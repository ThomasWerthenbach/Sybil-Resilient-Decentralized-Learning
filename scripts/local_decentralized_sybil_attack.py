import os
from typing import List

from ipv8.peer import Peer

from experiment_infrastructure.decentralized_learning.node_manager import NodeManager
from experiment_infrastructure.experiment_settings.settings import Settings


class MockPeer:
    pass


if __name__ == '__main__':
    filename = os.path.join(os.path.dirname(__file__), 'settings2.json')
    with open(filename) as f:
        s = Settings.from_json("".join([x.strip() for x in f.readlines()]))

    hosts: List[NodeManager] = list()
    for i in range(s.total_hosts):
        hosts.append(NodeManager(s,
                                 i + 1,
                                 MockPeer(),
                                 lambda self, peer, info, model: hosts[peer - 1].receive_model(self, info, model),
                                 lambda x, y: print(x, y)))

    # Kickstart
    for host in hosts:
        print("START EPOCH")
        host.start_next_epoch()
