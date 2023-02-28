from experiment_infrastructure.attacks.attack import Attack
from ml.datasets.partitioner import Partition


class LabelFlip(Attack):
    def __init__(self, f: int, t: int):
        self.f = f
        self.t = t

    def transform_data(self, data: Partition) -> Partition:
        d = list()
        for i in range(len(data.data)):
            x, y = data.data[i]
            if y == self.f:
                d.append((x, self.t))
            else:
                d.append((x, y))
        return Partition(d, data.index)
