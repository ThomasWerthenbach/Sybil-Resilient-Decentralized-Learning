import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from ml.datasets.dataset import Dataset
from ml.datasets.partitioner import DataPartitioner


class EMNISTLetterDataset(Dataset):
    NUM_CLASSES = 27

    def all_training_data(self, batch_size=32, shuffle=False):
        return DataLoader(torchvision.datasets.EMNIST(
            root=self.DEFAULT_DATA_DIR + '/train', train=True, download=True, transform=ToTensor(), split='letters'
        ), batch_size=batch_size, shuffle=shuffle)

    def get_peer_dataset(self, peer_id: int, total_peers: int, non_iid=False, sizes=None, batch_size=32, shuffle=False):
        if sizes is None:
            sizes = [1.0 / total_peers for _ in range(total_peers)]
        # sizes = [0.001, 0.001]
        data = torchvision.datasets.EMNIST(
            root=self.DEFAULT_DATA_DIR + '/train', train=True, download=True, transform=ToTensor(), split='letters'
        )
        return DataLoader(DataPartitioner(data, sizes).use(peer_id), batch_size=batch_size, shuffle=shuffle)

    def all_test_data(self, batch_size=32, shuffle=False):
        return DataLoader(torchvision.datasets.EMNIST(
            root=self.DEFAULT_DATA_DIR + '/test', train=False, download=True, transform=ToTensor(), split='letters'
        ))
