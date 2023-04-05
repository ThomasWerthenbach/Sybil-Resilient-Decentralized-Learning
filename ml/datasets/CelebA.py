import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from experiment_infrastructure.attacks.attack import Attack
from ml.datasets.dataset import Dataset
from ml.datasets.partitioner import DataPartitioner

IMAGE_DIM = 84
CHANNELS = 3
NUM_CLASSES = 2
TRAIN_FILE = "train/all_data_niid_1_keep_3_train_9.json"
TEST_FILE = "test/all_data_niid_1_keep_3_test_9.json"

class Celeba(Dataset):

    def all_training_data(self, batch_size=32, shuffle=False) -> DataLoader:
        _, _, train_data = self.__read_file__(
            os.path.join(self.DEFAULT_DATA_DIR, TRAIN_FILE))

        all_train_data = []
        all_train_data_labels = []
        for _, v in train_data.items():
            all_train_data.extend(v["x"])
            all_train_data_labels.extend(v["y"])
        all_train_data = self.process_x(all_train_data)
        all_train_data_labels = np.array(all_train_data_labels)
        return DataLoader(list(zip(all_train_data, all_train_data_labels)), batch_size=batch_size, shuffle=shuffle)

    def get_peer_dataset(self, peer_id: int, total_peers: int, non_iid=False, sizes=None, batch_size=8, shuffle=False,
                         sybil_data_transformer: Attack = None) -> DataLoader:
        _, _, train_data = self.__read_file__(
            os.path.join(self.DEFAULT_DATA_DIR, "train/all_data_niid_1_keep_3_train_9.json"))

        if non_iid:
            # todo support dirichlet-based distribution as well
            all_users = sorted(list(train_data.keys()))
            user_peer_id = all_users[peer_id]
            peer_train_data = train_data[user_peer_id]["x"]
            peer_train_data_labels = train_data[user_peer_id]["y"]
            all_train_data = self.process_x(peer_train_data)
            all_train_data_labels = np.array(peer_train_data_labels)
            train_set = list(zip(all_train_data, all_train_data_labels))
        else:
            peer_train_data = []
            peer_train_data_labels = []
            for i in range(peer_id, len(train_data), total_peers):
                peer_train_data.extend(train_data.values()[i]["x"])
                peer_train_data_labels.extend(train_data.values()[i]["y"])
            pre_train_set = DataPartitioner(list(zip(peer_train_data, peer_train_data_labels)), sizes).use(peer_id)
            train_x = []
            train_y = []
            for i in range(len(pre_train_set)):
                train_x.append(pre_train_set[i][0])
                train_y.append(pre_train_set[i][1])
            train_set = list(zip(self.process_x(train_x), train_y))
        train = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

        return train

    def all_test_data(self, batch_size=32, shuffle=False) -> DataLoader:
        _, _, test_data = self.__read_file__(
            os.path.join(self.DEFAULT_DATA_DIR, TEST_FILE))
        all_test_data = []
        all_test_data_labels = []
        for _, v in test_data.items():
            all_test_data.extend(v["x"])
            all_test_data_labels.extend(v["y"])
        all_test_data = self.process_x(all_test_data)
        all_test_data_labels = np.array(all_test_data_labels)
        test = DataLoader(list(zip(all_test_data, all_test_data_labels)), batch_size=120, shuffle=True)

        return test

    def __read_file__(self, file_path):
        with open(file_path, "r") as inf:
            client_data = json.load(inf)
        return (
            client_data["users"],
            client_data["num_samples"],
            client_data["user_data"],
        )

    # def __read_dir__(self, data_dir):
    #
    #     clients = []
    #     num_samples = []
    #     data = defaultdict(lambda: None)
    #
    #     files = os.listdir(data_dir)
    #     files = [f for f in files if f.endswith(".json")]
    #     for f in files:
    #         file_path = os.path.join(data_dir, f)
    #         u, n, d = self.__read_file__(file_path)
    #         clients.extend(u)
    #         num_samples.extend(n)
    #         data.update(d)
    #     return clients, num_samples, data

    # def file_per_user(self, dir, write_dir):
    #     clients, num_samples, train_data = self.__read_dir__(dir)
    #     for index, client in enumerate(clients):
    #         my_data = dict()
    #         my_data["users"] = [client]
    #         my_data["num_samples"] = num_samples[index]
    #         my_samples = {"x": train_data[client]["x"], "y": train_data[client]["y"]}
    #         my_data["user_data"] = {client: my_samples}
    #         with open(os.path.join(write_dir, client + ".json"), "w") as of:
    #             json.dump(my_data, of)
    #             print("Created File: ", client + ".json")

    # def load_trainset(self):
    #     """
    #     Loads the training set. Partitions it if needed.
    #
    #     """
    #     logging.info("Loading training set.")
    #     files = os.listdir(self.train_dir)
    #     files = [f for f in files if f.endswith(".json")]
    #     files.sort()
    #     c_len = len(files)
    #
    #     # clients, num_samples, train_data = self.__read_dir__(self.train_dir)
    #
    #     if self.sizes == None:  # Equal distribution of data among processes
    #         e = c_len // self.n_procs
    #         frac = e / c_len
    #         self.sizes = [frac] * self.n_procs
    #         self.sizes[-1] += 1.0 - frac * self.n_procs
    #         logging.debug("Size fractions: {}".format(self.sizes))
    #
    #     self.uid = self.mapping.get_uid(self.rank, self.machine_id)
    #     my_clients = DataPartitioner(files, self.sizes).use(self.uid)
    #     my_train_data = {"x": [], "y": []}
    #     self.clients = []
    #     self.num_samples = []
    #     logging.debug("Clients Length: %d", c_len)
    #     logging.debug("My_clients_len: %d", my_clients.__len__())
    #     for i in range(my_clients.__len__()):
    #         cur_file = my_clients.__getitem__(i)
    #
    #         clients, _, train_data = self.__read_file__(
    #             os.path.join(self.train_dir, cur_file)
    #         )
    #         for cur_client in clients:
    #             logging.debug("Got data of client: {}".format(cur_client))
    #             self.clients.append(cur_client)
    #             my_train_data["x"].extend(self.process_x(train_data[cur_client]["x"]))
    #             my_train_data["y"].extend(train_data[cur_client]["y"])
    #             self.num_samples.append(len(train_data[cur_client]["y"]))
    #
    #     logging.debug(
    #         "Initial shape of x: {}".format(
    #             np.array(my_train_data["x"], dtype=np.dtype("float32")).shape
    #         )
    #     )
    #     self.train_x = (
    #         np.array(my_train_data["x"], dtype=np.dtype("float32"))
    #         .reshape(-1, IMAGE_DIM, IMAGE_DIM, CHANNELS)
    #         .transpose(0, 3, 1, 2)  # Channel first: torch
    #     )
    #     self.train_y = np.array(my_train_data["y"], dtype=np.dtype("int64")).reshape(-1)
    #     logging.debug("train_x.shape: %s", str(self.train_x.shape))
    #     logging.debug("train_y.shape: %s", str(self.train_y.shape))
    #     assert self.train_x.shape[0] == self.train_y.shape[0]
    #     assert self.train_x.shape[0] > 0
    #
    # def load_testset(self):
    #     """
    #     Loads the testing set.
    #
    #     """
    #     logging.info("Loading Celeba testing set.")
    #     _, _, d = self.__read_dir__(self.test_dir)
    #     test_x = []
    #     test_y = []
    #     for test_data in d.values():
    #         test_x.extend(self.process_x(test_data["x"]))
    #         test_y.extend(test_data["y"])
    #     self.test_x = (
    #         np.array(test_x, dtype=np.dtype("float32"))
    #         .reshape(-1, IMAGE_DIM, IMAGE_DIM, CHANNELS)
    #         .transpose(0, 3, 1, 2)
    #     )
    #     self.test_y = np.array(test_y, dtype=np.dtype("int64")).reshape(-1)
    #     logging.debug("test_x.shape: %s", str(self.test_x.shape))
    #     logging.debug("test_y.shape: %s", str(self.test_y.shape))
    #     assert self.test_x.shape[0] == self.test_y.shape[0]
    #     assert self.test_x.shape[0] > 0

    def process_x(self, raw_x_batch):
        """
        Preprocesses the whole batch of images

        Returns
        -------
        np.array
            The images as a numpy array

        """
        x_batch = [self._load_image(i) for i in raw_x_batch]
        return np.array(x_batch, dtype=np.dtype("float32")).reshape(-1, IMAGE_DIM, IMAGE_DIM, CHANNELS).transpose(0, 3, 1, 2)

    def _load_image(self, img_name):
        """
        Open and load image.

        Returns
        -------
        np.array
            The image as a numpy array

        """
        img = Image.open(os.path.join(self.DEFAULT_DATA_DIR, "train/img_align_celeba", img_name))
        img = img.resize((IMAGE_DIM, IMAGE_DIM)).convert("RGB")
        return np.array(img)


if __name__ == '__main__':
    celeba = Celeba()
    celeba.do_thing()
