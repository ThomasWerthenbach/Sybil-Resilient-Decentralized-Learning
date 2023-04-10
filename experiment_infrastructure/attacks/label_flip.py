import random
from typing import List, Dict

from torch.utils.data import DataLoader

from experiment_infrastructure.attacks.attack import Attack
from experiment_infrastructure.experiment_settings.settings import Settings
from ml.datasets.partitioner import Partition, DataPartitioner


class LabelFlip(Attack):
    def transform_eval_data(self, eval_data: DataLoader):
        attack_data = list()
        for x, y in eval_data.dataset:
            if y == self.f:
                attack_data.append((x, self.t))
            elif y == self.t:
                attack_data.append((x, self.f))
        return DataLoader(attack_data, batch_size=120, shuffle=False)

    def __init__(self, settings: Settings, f: int = 0, t: int = 1, seed=42):
        self.f = f
        self.t = t
        self.seed = seed

    def transform_data(self, data: Partition, trainset: Dict[object, List], sizes, peer_id) -> List:
        all_targeted_train_data = list(map(lambda x: (x, self.t), trainset[self.f])) + list(map(lambda x: (x, self.f), trainset[self.t]))

        transformed_data = list()
        for i in range(len(data)):
            x, y = data[i]
            if y == self.f:
                transformed_data.append((x, self.t))
            elif y == self.t:
                transformed_data.append((x, self.f))
            else:
                transformed_data.append((x, y))

        all_data = transformed_data + all_targeted_train_data
        random.seed(self.seed)
        random.shuffle(all_data)

        return all_data
