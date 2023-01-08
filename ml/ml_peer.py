from community.ml_community import MLCommunity
from experiment_settings.algorithms import Algorithm
from experiment_settings.settings import Settings
from ml.executor.executor import Executor


class MLPeer:
    """
    This class controls the peer's main life cycle.

    The main life cycle consists of:
    1: peer created
    2: train 1 epoch
    3: model trained
    4: send weights to existing peers
    5: goto step 9 if algorithm is not Repple
    6: check if new peers have been discovered, if so, send weights and a data to them. DEPENDS ON STRANGER POLICY
    7: process received connection requests (verify them and give them a score between 0 and 1).
       DEPENDS ON REPUTATION
    8: By now, most peers should have finished training, and we have received their weights.
    9: Evaluate the new weights and update their score and perform the weight integration.
    10: GOTO 2
    """

    def __init__(self, peer_id: int, settings: Settings):
        self.peer_id = peer_id
        self.settings = settings

    def start_lifecycle(self, community: MLCommunity):
        """
        Method called by ipv8 overlay when the peer is ready to start.
        """
        print(self.peer_id, "started")
        executor: Executor = Executor.get_executor_class(self.settings.algorithm)(self.settings)

        executor.load_data(self.peer_id, self.settings.total_peers, self.settings.non_iid)
        executor.prepare_model()

        # todo: datadistributor. should be able to get a sample to send in introductions (if still necessary in Repple)

        for _ in range(self.settings.max_rounds):
            print("start trainging")
            executor.train()
            print("finished training")

            weights = executor.get_model_weights()
            print("sending weights")
            community.send_model_to_peers(weights)
            print("sent weights")
            if self.settings.algorithm == Algorithm.REPPLE:
                # todo: send introduction requests
                # todo: process received introduction requests, based on stranger policy
                pass
            print("start integrating")
            other_models = community.get_other_models()
            if len(other_models) > 0:
                prioritized_models = executor.prioritize_other_models(other_models)
                executor.integrate_models(prioritized_models)

            print("done integrating")
