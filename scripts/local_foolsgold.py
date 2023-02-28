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
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from experiment_infrastructure.attacks.label_flip import LabelFlip
from experiment_infrastructure.experiment_settings.settings import Settings
from experiment_infrastructure.federated_learning.node_manager import NodeManager
from ml.aggregators.foolsgold import FoolsGold
from ml.aggregators.util import weighted_average
from ml.datasets.MNIST import MNISTDataset
from ml.util import deserialize_model, serialize_model, model_difference, model_sum

models =defaultdict(dict)
previous_models = defaultdict(dict)
accumulated_update_history = defaultdict(dict)

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


if __name__ == '__main__':
    filename = os.path.join(os.path.dirname(__file__),
                            '..\\gumby\\experiments\\FL_IID_AVG_MNIST_SYBIL_LABEL_FLIP\\settings.json')
    with open(filename) as f:
        s = Settings.from_json("".join([x.strip() for x in f.readlines()]))

    print(f"Initializing node managers of size {1.0 / s.total_hosts}")
    hosts: List[NodeManager] = list()
    for i in range(s.total_hosts):
        hosts.append(NodeManager(s, i + 1, send_model))

    foolsgold = FoolsGold()

    data = torchvision.datasets.MNIST(
        root='C:\\Users\\takwe\\tu\\Repple\\data\\test', train=False, download=True, transform=ToTensor()
    )
    test_data = DataLoader(data, batch_size=120, shuffle=False)

    new_data = list()
    attack = LabelFlip()
    for x, y in data:
        if y == attack.f:
            new_data.append((x, attack.t))

    attack_rate_data = DataLoader(new_data, batch_size=120, shuffle=False)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(device_name)
    device = torch.device(device_name)

    round = 0

    for host in hosts:
        print("next host")
        host.start_next_epoch()

    for _ in range(25):
        round += 1

        print("TRAINING DONE ================================ CALCULATING ATTACK RATE")
        print("Models has a length of: ", len(models))

        # average_model = weighted_average(models, [1.0 / len(models) for _ in range(len(models))])

        peers = list(models.keys())
        m = reduce(add, map(lambda x: list(models[x].values()), peers))
        h = reduce(add, map(lambda x: list(accumulated_update_history[x].values()), peers))
        average_model = foolsgold.aggregate(m, h)
        previous_models = copy.deepcopy(models)
        models.clear()

        average_model.eval()
        test_loss = 0
        test_corr = 0
        average_model.to(device)
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(device), target.to(device)
                output = average_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_corr += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_data)
        test_acc = 100. * test_corr / (len(test_data) * 120)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, test_corr, len(test_data) * 120, test_acc))

        test_loss = 0
        test_corr = 0
        with torch.no_grad():
            for data, target in attack_rate_data:
                data, target = data.to(device), target.to(device)
                output = average_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                test_corr += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(attack_rate_data)
        test_acc = 100. * test_corr / (len(attack_rate_data) * 120)
        print('Attack rate test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, test_corr, len(attack_rate_data) * 120, test_acc))

        for host in hosts:
            print("next host")
            host.receive_model(None, json.dumps({'round': round}).encode(), serialize_model(average_model))
