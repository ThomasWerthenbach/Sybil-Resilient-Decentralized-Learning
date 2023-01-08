
import logging

import torch

from datasets.FashionMNIST import FashionMNISTDataset
from ml.models.FashionMNIST import FashionMNISTCNN

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = FashionMNISTCNN()
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.005)
    dataset = FashionMNISTDataset()
    trainloader = dataset.get_peer_dataset(batch_size=120, shuffle=True)
    testloader = dataset.all_test_data(batch_size=100, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    testloader = dataset.get_peer_dataset()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 train images: %d %%' % (
        100 * correct / total))

    for p in model.parameters():
        print(p)


# data_dir = os.path.join(os.environ["HOME"], "leaf", "femnist")
# train_dir = os.path.join(data_dir, "per_user_data/train")
# test_dir = os.path.join(data_dir, "data/test")
#
# mapping = Linear(1, 100)
# s = Femnist(0, 0, mapping, train_dir=train_dir, test_dir=test_dir)
#
# print("Datasets prepared")
#
# # Model
# model = create_model(settings.dataset)
# print(model)
#
# print("Initial evaluation")
# print(s.test(model))
#
# for round in range(NUM_ROUNDS):
#     start_time = time.time()
#     print("Starting training round %d" % (round + 1))
#     trainer = ModelTrainer(data_dir, settings, 0)
#     trainer.train(model)
#     print("Training round %d done - time: %f" % (round + 1, time.time() - start_time))
#     print(s.test(model))
