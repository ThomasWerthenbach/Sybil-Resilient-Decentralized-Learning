
import logging

from ml.transfer_settings import TransferSettings
from transfer_scripts.TransferLearner import TransferLearner

logging.basicConfig(level=logging.INFO)

learning_settings = TransferSettings(
    learning_rate=0.001,
    dataset="mnist",
    model="mnist",
    train_shuffle=True,
    test_shuffle=False,
    train_batch_size=120,
    test_batch_size=100,
    epochs=10
)

if __name__ == '__main__':
    TransferLearner(learning_settings).train()
