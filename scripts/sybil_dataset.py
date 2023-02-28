from experiment_infrastructure.attacks.label_flip import LabelFlip
from ml.datasets.MNIST import MNISTDataset

if __name__ == "__main__":
    a = MNISTDataset().get_peer_dataset(1, 100, sybil_data_transformer=LabelFlip(0, 1))
    b = MNISTDataset().get_peer_dataset(1, 100)
    al = list(a)
    bl = list(b)
    for first, second in zip(al, bl):
        print(first[1], "\t", second[1])
