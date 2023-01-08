from typing import List, Dict

from ipv8.peer import Peer
from torch import Tensor

from ml.executor.executor import Executor
from ml.integrators.average import AverageIntegrator
from ml.prioritizers.average import AveragePrioritizer


class AverageExecutor(Executor):

    def __init__(self, settings):
        super().__init__(settings)
        self.prioritizer = AveragePrioritizer()
        self.integrator = AverageIntegrator()
        self.dataset = None

    def get_model_weights(self) -> List[List[str]]:
        return self.model.get_serialized_layer_weights()

    def prioritize_other_models(self, other_models: Dict[Peer, List[List[float]]]) -> List[Tensor]:
        return self.prioritizer.filter_models(self.model.get_output_layer_weights(),
                                              list(map(Tensor, other_models.values())))

    def integrate_models(self, prioritized_models: List[Tensor]):
        round_result: Tensor = self.integrator.integrate(self.model.get_output_layer_weights(), prioritized_models)
        self.model.replace_output_layer(round_result)
