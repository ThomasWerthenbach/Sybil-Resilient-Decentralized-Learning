from community.ml_community import MLCommunity
from community.settings import Settings
from ml.executor.executor import Executor, Algorithm


class MLPeer:
    """
    This class controls the peer's main life cycle.

    The main life cycle consists of:
    1: peer created
    2: record time
    3: train first epoch
    4: model trained
    5: send weights to existing peers
    6: goto step 9 if algorithm is not Repple
    7: check if new peers have been discovered, if so, send weights and a data to them. DEPENDS ON STRANGER POLICY
    8: process received connection requests (verify them and give them a score between 0 and 1).
       DEPENDS ON REPUTATION
    9: Wait until X time has passed since recorded timestamp.
    10: By now, most peers should have finished training, and we have received their weights.
    11: Evaluate the new weights and update their score and perform the weight integration.
    12: GOTO 2
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def start_lifecycle(self, community: MLCommunity):
        """
        Method called by ipv8 overlay when the peer is ready to start.
        """
        executor: Executor = Executor.get_executor_class(self.settings.algorithm)(self.settings)

        executor.prepare_model()

        # todo: create a data distributor. should be able to provide the data class-wise and get a sample to send in introductions

        for _ in range(self.settings.max_rounds):
            executor.train()

            weights = executor.get_model_weights()
            # Stringify all floats
            weights = list(map(lambda x: list(map(str, x)), weights))
            community.send_model_to_peers(weights)

            if self.settings.algorithm == Algorithm.REPPLE:
                # todo: send introduction requests
                # todo: process received introduction requests, based on stranger policy
                pass

            # todo wait for some time?
            other_models = community.get_other_models()
            prioritized_models = executor.prioritize_other_models(other_models)
            executor.integrate_models(prioritized_models)
