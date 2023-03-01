import random
from typing import List, Dict

from experiment_infrastructure.attacks.attack import Attack
from ml.datasets.partitioner import Partition, DataPartitioner


class LabelFlip(Attack):
    def __init__(self, f: int = 0, t: int = 1, seed=42):
        self.f = f
        self.t = t
        self.seed = seed

    def transform_data(self, data: Partition, trainset: Dict[object, List], sizes, peer_id) -> Partition:
        all_targeted_train_data = list(map(lambda x: (x, self.t), trainset[self.f]))

        transformed_data = list()
        for i in range(len(data)):
            x, y = data[i]
            if y == self.f:
                transformed_data.append((x, self.t))
            else:
                transformed_data.append((x, y))

        all_data = transformed_data + all_targeted_train_data
        random.seed(self.seed)
        random.shuffle(all_data)

        return DataPartitioner(all_data, sizes).use(peer_id)
