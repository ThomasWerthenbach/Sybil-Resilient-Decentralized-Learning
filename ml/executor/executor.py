import logging
from abc import ABC, abstractmethod
from typing import Type, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from ipv8.peer import Peer

from datasets.dataset import Dataset
from experiment_settings.algorithms import Algorithm
from experiment_settings.settings import Settings
from ml.models.model import Model


class Executor(ABC):
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = Model.get_model_class(self.settings.model)()
        self.peer_dataset = None
        self.test_dataset = None
        self.num_classes = None

    @staticmethod
    def get_executor_class(executor_name: Algorithm) -> Type['Executor']:
        if executor_name == Algorithm.AVERAGE:
            from ml.executor.average import AverageExecutor
            return AverageExecutor
        else:
            raise ValueError("Unknown executor: " + executor_name.value)

    def prepare_model(self):
        """
        Load pretrained model, replace output layer and freeze all non-output layers.
        """
        self.model.load_state_dict(torch.load(self.settings.pretrained_model_location))
        self.model.prepare_for_transfer_learning(self.num_classes)

    def train(self):
        """
        Train the model for one epoch
        """
        if self.peer_dataset is None:
            raise RuntimeError("No peer dataset available")
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.settings.learning_rate)

        self.model.train()
        train_loss = 0
        train_corr = 0
        for i, data in enumerate(self.peer_dataset, 0):
            inputs, labels = data
            inputs, labels = inputs.to('cpu'), labels.to('cpu')
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = F.nll_loss(outputs, labels)
            train_pred = outputs.argmax(dim=1, keepdim=True)
            train_corr += train_pred.eq(labels.view_as(train_pred)).sum().item()
            train_loss += F.nll_loss(outputs, labels, reduction='sum').item()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                logging.info('Train Epoch status [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, len(self.peer_dataset), 100. * i / len(self.peer_dataset), loss.item()))
        # self.model.eval()
        # test_loss = 0
        # correct = 0
        # total_pred = np.zeros(0)
        # total_target = np.zeros(0)
        # with torch.no_grad():
        #     for data, target in self.test_dataset:
        #         data, target = data.to('cpu'), target.to('cpu', dtype=torch.int64)
        #         output = self.model(data)
        #         test_loss += F.nll_loss(output, target, reduction='sum').item()
        #         pred = output.argmax(dim=1, keepdim=True)
        #         total_pred = np.append(total_pred, pred.cpu().numpy())
        #         total_target = np.append(total_target, target.cpu().numpy())
        #         correct += pred.eq(target.view_as(pred)).sum().item()
        #     logging.info("Epoch done! correct images: %5f" % (correct / (len(self.test_dataset) * len(target))))

    @abstractmethod
    def get_model_weights(self) -> List[List[str]]:
        """
        Get the weights of the output layer of the trained model
        """

    @abstractmethod
    def prioritize_other_models(self, other_models: Dict[Peer, List[List[float]]]) -> List[List[List[float]]]:
        """
        Prioritize/filter other models based on algorithm-specific criteria
        """

    @abstractmethod
    def integrate_models(self, prioritized_models):
        """
        Integrate the prioritized models into the local model
        """

    def load_data(self, peer_id: int, total_peers: int, non_iid: bool = False):
        """
        Load the data assigned to this peer
        """
        self.peer_dataset = Dataset.get_dataset_class(self.settings.model)().get_peer_dataset(peer_id, total_peers,
                                                                                              non_iid, sizes=None)
        self.test_dataset = Dataset.get_dataset_class(self.settings.model)().all_test_data()
        self.num_classes = Dataset.get_dataset_class(self.settings.model).NUM_CLASSES
