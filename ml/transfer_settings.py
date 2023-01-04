class TransferSettings:
    """
    Settings used for training a model on a single machine to later be used for transfer learning.
    """

    def __init__(self, learning_rate: float,
                 dataset: str,
                 model: str,
                 train_shuffle: bool = False,
                 test_shuffle: bool = False,
                 train_batch_size: int = 32,
                 test_batch_size: int = 32,
                 epochs: int = 10):
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.model = model
        self.train_shuffle = train_shuffle
        self.test_shuffle = test_shuffle
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
