import time

import numpy as np
from ipv8.community import Community

from ml.models.EMNIST import EMNIST
import sklearn.metrics.pairwise as smp


class CosineEvalCommunity(Community):
    """
    Main cosine evaluation community.
    """
    community_id = b"\x08\x07\xee\xa6\x97OgZ\xa1\x19\x14o\xfc\x87\xbe\x16/'Ka"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experiment_module = None
        self.peer_id = 0

    def started(self):
        pass

    def start_cosine_exec(self, peer_id, experiment_module):
        self.peer_id = peer_id
        self.experiment_module = experiment_module
        self.experiment_module.autoplot_create(f"cosine_eval_peers")
        self.experiment_module.autoplot_create(f"cosine_eval_time")
        self.register_task("start_lifecycle_" + str(peer_id), self.start_lifecycle, delay=0)

    def get_peer_amount(self, round_number: int) -> int:
        """
        Generates number of peers to simulate.

        For node with id 1, the sequence will be:
        2500
        5000
        7500
        10000

        For node with id 2, the sequence will be:
        2750
        5250
        7750
        10250

        Suitable for 10 nodes.
        """
        return 2500 * round_number + 250 * (self.peer_id - 1)  # peer_id is one-indexed

    async def start_lifecycle(self):
        model_parameters = filter(lambda p: p.requires_grad, EMNIST().parameters())
        amount_of_param = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f"amount of trainable parameters: {amount_of_param}")

        r = 0
        while True:
            r += 1

            # Create dummy data
            models = list()
            for _ in range(self.get_peer_amount(r)):
                models.append(np.random.rand(amount_of_param))

            # Perform execution
            start = time.time()
            smp.cosine_similarity(models)
            end = time.time()

            # Log results
            self.experiment_module.autoplot_add_point("cosine_eval_peers", self.get_peer_amount(r))
            self.experiment_module.autoplot_add_point("cosine_eval_time", end - start)

            self.logger.info(f"Done with {self.get_peer_amount(r)} peers and it took {end - start} seconds")
