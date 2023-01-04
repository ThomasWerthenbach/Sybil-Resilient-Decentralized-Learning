import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from datasets.dataset import Dataset


class EMNISTLetterDataset(Dataset):
    def load_trainset(self, batch_size=32, shuffle=False):
        return DataLoader(torchvision.datasets.EMNIST(
            root=self.DEFAULT_DATA_DIR + '/train', train=True, download=True, transform=ToTensor(), split='letters'
        ), batch_size=batch_size, shuffle=shuffle)

    def load_testset(self, batch_size=32, shuffle=False):
        return DataLoader(torchvision.datasets.EMNIST(
            root=self.DEFAULT_DATA_DIR + '/test', train=False, download=True, transform=ToTensor(), split='letters'
        ), batch_size=batch_size, shuffle=shuffle)
