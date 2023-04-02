import os

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from experiment_infrastructure.attacks.backdoor import Backdoor
from experiment_infrastructure.attacks.label_flip import LabelFlip
from experiment_infrastructure.experiment_settings.settings import Settings
from ml.datasets.MNIST import MNISTDataset

if __name__ == "__main__":
    # d = DataLoader(torchvision.datasets.MNIST(
    #     root=os.path.join(os.path.dirname(__file__), '../data', 'train'), train=True, download=True,
    #     transform=ToTensor()
    # ), batch_size=1)

    filename = os.path.join(os.path.dirname(__file__), 'settings.json')
    with open(filename) as f:
        s = Settings.from_json("".join([x.strip() for x in f.readlines()]))

    d = MNISTDataset().get_peer_dataset(0, s.peers_per_host * s.total_hosts, sybil_data_transformer=Backdoor(s))

    for data, target in d:
        print(data[0][0])
        # print(data.shape)
        print(target)
        break
