import time

import numpy as np
from ipv8.community import Community
from numpy.linalg import norm

from ml.models.CIFAR10 import LeNet


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
        return 2500 * round_number

    async def start_lifecycle(self):
        model_parameters = filter(lambda p: p.requires_grad, LeNet().parameters())
        amount_of_param = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f"amount of trainable parameters: {amount_of_param}")

        # Generate two random models
        model1 = np.random.rand(amount_of_param)
        model2 = np.random.rand(amount_of_param)

        r = 0
        while True:
            r += 1

            # Perform execution -> we simulate pairwise cosine similarity
            start = time.time()
            for _ in range(self.get_peer_amount(r)):
                for _ in range(self.get_peer_amount(r)):
                    np.dot(model1, model2) / (norm(model1) * norm(model2))
            end = time.time()

            # Log results
            self.experiment_module.autoplot_add_point("cosine_eval_peers", self.get_peer_amount(r))
            self.experiment_module.autoplot_add_point("cosine_eval_time", end - start)

            self.logger.info(f"Done with {self.get_peer_amount(r)} peers and it took {end - start} seconds")
