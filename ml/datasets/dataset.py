import os
import logging
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader

from experiment_infrastructure.attacks.attack import Attack
from ml.datasets.partitioner import DataPartitioner, DirichletDataPartitioner


class Dataset(ABC):
    NUM_CLASSES = None
    DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../data')

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def all_training_data(self, batch_size=32, shuffle=False) -> DataLoader:
        """
        Retrieves the entire dataset (used for transfer learning preparations)
        """

    @abstractmethod
    def get_peer_dataset(self, peer_id: int, total_peers: int, non_iid=False, sizes=None, batch_size=8, alpha=0.1, shuffle=True,
                         sybil_data_transformer: Attack = None) -> DataLoader:
        """
        Function to load the training set
        """

    @abstractmethod
    def all_test_data(self, batch_size=32, shuffle=False) -> DataLoader:
        """
        Function to load the test set
        """

    def get_peer_train_set(self, data, peer_id, total_peers, non_iid, sizes, batch_size, alpha, shuffle,
                           sybil_data_transformer):
        self.logger.info(f"Initializing dataset of size {1.0 / total_peers} for peer {peer_id}. Non-IID: {non_iid}. Alpha: {alpha}")
        if sizes is None:
            sizes = [1.0 / total_peers for _ in range(total_peers)]
        if not non_iid:
            train_set = DataPartitioner(data, sizes).use(peer_id)
        else:
            train_set = DirichletDataPartitioner(
                data, sizes, alpha=alpha
            ).use(peer_id)
        if sybil_data_transformer is not None:
            train_data = {key: [] for key in range(10)}
            for x, y in data:
                train_data[y].append(x)
            train_set = sybil_data_transformer.transform_data(train_set, train_data, sizes, peer_id)

        return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
