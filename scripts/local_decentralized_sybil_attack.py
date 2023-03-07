import copy
import json
import os
from collections import defaultdict
from functools import reduce
from operator import add
from typing import List

import torch
import torch.nn.functional as F
import torchvision
from ipv8.types import Peer
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from experiment_infrastructure.attacks.label_flip import LabelFlip
from experiment_infrastructure.experiment_settings.settings import Settings
from experiment_infrastructure.federated_learning.node_manager import NodeManager
from experiment_infrastructure.federated_learning.server_manager import ServerManager
from ml.aggregators.foolsgold import FoolsGold
from ml.aggregators.util import weighted_average
from ml.datasets.MNIST import MNISTDataset
from ml.util import deserialize_model, serialize_model, model_difference, model_sum

models = defaultdict(dict)
previous_models = defaultdict(dict)
accumulated_update_history = defaultdict(dict)
info: bytes | None = None
model: bytes | None = None


def send_model(info: bytes, serialized_model: bytes):
    info = json.loads(info.decode())
    p = info['peer']

    # Store model properly
    model = deserialize_model(serialized_model, s)
    models[host][p] = model

    # Calculate model difference
    if p in previous_models[host]:
        difference = model_difference(previous_models[host][p], model)
    else:
        difference = model

    # Store model updates properly
    if p in accumulated_update_history[host]:
        accumulated_update_history[host][p] = model_sum(accumulated_update_history[host][p], difference)
    else:
        accumulated_update_history[host][p] = difference


def server_distributes_model(peer: Peer, _info: bytes, _model: bytes):
    global info, model
    info = _info
    model = _model


if __name__ == '__main__':
    filename = os.path.join(os.path.dirname(__file__),
                            '..\\gumby\\experiments\\FL_IID_AVG_MNIST_SYBIL_LABEL_FLIP\\settings.json')
    with open(filename) as f:
        s = Settings.from_json("".join([x.strip() for x in f.readlines()]))

    server = ServerManager(s, server_distributes_model, lambda x, y: print(x, y))

    hosts: List[NodeManager] = list()
    for i in range(s.total_hosts):
        hosts.append(NodeManager(s, i + 1, lambda _info, _model: server.receive_model(0, _info, _model)))

    # Kickstart
    for host in hosts:
        host.start_next_epoch()

    for i in range(25):
        assert info is not None
        assert model is not None
        print("Start round", i)
        for host in hosts:
            host.receive_model(None, info, model)
