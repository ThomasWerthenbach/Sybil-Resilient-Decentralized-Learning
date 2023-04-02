import logging

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, utils
from PIL import Image
import os.path
import torch
from torchvision.transforms import ToTensor

from experiment_infrastructure.attacks.attack import Attack
from ml.datasets.dataset import Dataset
from ml.datasets.partitioner import DataPartitioner, DirichletDataPartitioner


class FEMNIST(MNIST):
    """
    Code adopted from https://github.com/tao-shen/FEMNIST_pytorch
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        ('https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
         '59c65cec646fc57fe92d27d83afdf0ed')]

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        if not os.path.exists(self.raw_folder):
            os.makedirs(self.raw_folder)
        if not os.path.exists(self.processed_folder):
            os.makedirs(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)


class FEMNISTDataset(Dataset):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def all_training_data(self, batch_size=32, shuffle=False) -> DataLoader:
        return DataLoader(FEMNIST(
            root=self.DEFAULT_DATA_DIR + '/train', train=True, download=True, transform=ToTensor(),
        ), batch_size=batch_size, shuffle=shuffle)

    def get_peer_dataset(self, peer_id: int, total_peers: int, non_iid=False, sizes=None, batch_size=20, shuffle=False,
                         sybil_data_transformer: Attack = None):
        self.logger.info(f"Initializing dataset of size {1.0 / total_peers} for peer {peer_id}. Non-IID: {non_iid}")
        if sizes is None:
            sizes = [1.0 / total_peers for _ in range(total_peers)]
        data = FEMNIST(
            root=self.DEFAULT_DATA_DIR + '/train', train=True, download=True, transform=ToTensor(),
        )
        if not non_iid:
            train_set = DataPartitioner(data, sizes).use(peer_id)
        else:
            train_set = DirichletDataPartitioner(
                data, sizes
            ).use(peer_id)
        if sybil_data_transformer is not None:
            train_data = {key: [] for key in range(10)}
            for x, y in data:
                train_data[y].append(x)
            train_set = sybil_data_transformer.transform_data(train_set, train_data, sizes, peer_id)

        return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

    def all_test_data(self, batch_size=32, shuffle=False):
        return DataLoader(FEMNIST(
            root=self.DEFAULT_DATA_DIR + '/test', train=False, download=True, transform=ToTensor()
        ), batch_size=batch_size, shuffle=shuffle)
