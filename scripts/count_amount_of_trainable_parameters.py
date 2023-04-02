import numpy as np

from ml.models.CIFAR10LENET import LeNet
from ml.models.Celeba import CNN

if __name__ == '__main__':

    for model in [CNN(), LeNet()]:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        print(sum([np.prod(p.size()) for p in model_parameters]))
