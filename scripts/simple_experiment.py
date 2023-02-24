from experiments.experiment_settings.settings import Settings
from experiments.federated_learning.node_manager import NodeManager


def done_training(peer, i, m, n: NodeManager):
    n.receive_model(peer, i, m)

if __name__ == '__main__':
    filename = '/home/thomas/tu/rp/repple/gumby/experiments/FL_IID_AVG_MNIST/settings.json'
    with open(filename) as f:
        s = Settings.from_json("".join([x.strip() for x in f.readlines()]))
    n = None
    n = NodeManager(s, 25, "myself", lambda p, i, m: done_training(p, i, m, n), "server", lambda k, v: print(f"{k}: {v}"))
    n.start_next_epoch()
