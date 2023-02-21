from experiment_settings.settings import Settings
from federated_learning.node_manager import NodeManager
from ml.models.model import Model

def done_training(peer, i, m, n: NodeManager):
    n.receive_model(peer, i, m)

if __name__ == '__main__':
    s = Settings(10)
    n = None
    n = NodeManager(s, 2, "myself", lambda p, i, m: done_training(p, i, m, n), "server", lambda k, v: print(f"{k}: {v}"))
    n.start_next_epoch()
