from community.ml_community import MLCommunity
from community.settings import Settings
from ml.models.model import Model


class MLPeer:
    """
    This class controls the peer's main life cycle.

    The main life cycle consists of:
    1: peer created
    2: record time
    3: train first epoch
    4: model trained
    5: send weights to existing peers
    6: check if new peers have been discovered, if so, send weights and a data to them. DEPENDS ON STRANGER POLICY
    7: process received connection requests (verify them and give them a score between 0 and 1).
       DEPENDS ON REPUTATION
    8: Wait until X time has passed since recorded timestamp.
    9: By now, most peers should have finished training, and we have received their weights.
    10: Evaluate the new weights and update their score and perform the weight integration.
    11: GOTO 2
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def start_lifecycle(self, community: MLCommunity):
        """
        Method called by ipv8 overlay when the peer is ready to start.
        """
        model = Model.get_model_class(self.settings.model)()
        # TODO: prepare model for transfer learning by freezing all layers and add a new fully connected layer.

        # todo: create a data distributor. should be able to provide the data class-wise and get a sample to send in introductions

        while True:
            # todo: train model for 1 epoch

            # todo: send weights to existing peers

            # todo: send introduction requests

            # todo: process received introduction requests

            # todo: Perform the selected weight aggregation method on the received weights
            pass

