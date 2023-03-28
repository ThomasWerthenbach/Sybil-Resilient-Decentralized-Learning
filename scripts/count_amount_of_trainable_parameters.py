import numpy as np
from torch import nn

from ml.models.CIFAR10CNN import ResNet8, ResNet14, ResNet56, ResNet44, ResNet20, ResNet26, ResNet32
from ml.models.model import Model
import torch.nn.functional as F
import torch.optim as optim
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)

if __name__ == '__main__':
    conf = {"in_channels": 3, "out_channels": 10}
    for model in [ResNet8(conf), ResNet14(conf), ResNet20(conf), ResNet26(conf), ResNet32(conf), ResNet44(conf), ResNet56(conf)]:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        print(sum([np.prod(p.size()) for p in model_parameters]))
