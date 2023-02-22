import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from datasets.partitioner import DataPartitioner
from experiment_settings.settings import Settings
from ml.aggregators.util import weighted_average
from ml.models.FasterMNIST import MNIST


def train(model: MNIST, dataset: DataLoader, settings: Settings, test_dataset: DataLoader):
    """
    Train the model for one epoch
    """
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = model.to(device)

    if dataset is None:
        raise RuntimeError("No peer dataset available")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=settings.learning_rate,
        momentum=settings.momentum)

    model.train()
    train_loss = 0
    print(f"Training for 1 epoch with dataset length: {len(dataset)}")
    for i, data in enumerate(dataset):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.nll_loss(outputs, labels)
        train_loss += F.nll_loss(outputs, labels, reduction='sum').item()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('Train Epoch status [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                i, len(dataset), 100. * i / len(dataset), loss.item()))

    # Evaluate the model on the full test set for results
    model.eval()
    test_loss = 0
    test_corr = 0
    with torch.no_grad():
        for data, target in test_dataset:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            test_corr += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_dataset)
    test_acc = 100. * test_corr / (len(test_dataset) * 120)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, test_corr, len(test_dataset) * 120, test_acc))


if __name__ == '__main__':
    filename = '/home/thomas/tu/rp/repple/gumby/experiments/FL_IID_AVG/settings.json'
    with open(filename) as f:
        s = Settings.from_json("".join([x.strip() for x in f.readlines()]))

    print(f"Initializing dataset of size {1.0 / s.total_peers}")
    sizes = [1.0 / s.total_peers for _ in range(s.total_peers)]
    data = torchvision.datasets.MNIST(
        root='/home/thomas/tu/rp/repple/data/train', train=True, download=True, transform=ToTensor(),
    )
    partitioner = DataPartitioner(data, sizes)

    test_data = DataLoader(torchvision.datasets.MNIST(
            root='/home/thomas/tu/rp/repple/data/test', train=False, download=True, transform=ToTensor()
        ), batch_size=32, shuffle=False)


    models = list()
    for i in range(s.total_peers):
        models.append(MNIST())

    for _ in range(3):
        for i in range(s.total_peers):
            train(models[i], DataLoader(partitioner.use(i), batch_size=32, shuffle=False), s, test_data)

        print("TRAINING DONE ================================")

        average_model = weighted_average(models, sizes)

        for i in range(s.total_peers):
            models[i] = average_model

