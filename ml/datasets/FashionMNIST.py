import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from ml.datasets.dataset import Dataset


class FashionMNISTDataset(Dataset):
    def get_peer_dataset(self, batch_size=32, shuffle=True):
        return DataLoader(torchvision.datasets.FashionMNIST(
            root=self.DEFAULT_DATA_DIR + '/train', train=True, download=True, transform=ToTensor()
        ), batch_size=batch_size, shuffle=shuffle)

    def all_test_data(self, batch_size=32, shuffle=False):
        return DataLoader(torchvision.datasets.FashionMNIST(
            root=self.DEFAULT_DATA_DIR + '/test', train=False, download=True, transform=ToTensor()
        ), batch_size=batch_size, shuffle=shuffle)
