
import logging

from ml.models.model import ModelType
from transfer_scripts.transfer_settings import TransferSettings
from transfer_scripts.TransferLearner import TransferLearner

logging.basicConfig(level=logging.INFO)

learning_settings = TransferSettings(
    learning_rate=0.001,
    model=ModelType.MNIST,
    train_shuffle=True,
    test_shuffle=False,
    train_batch_size=120,
    test_batch_size=100,
    epochs=100
)

if __name__ == '__main__':
    TransferLearner(learning_settings).train()
