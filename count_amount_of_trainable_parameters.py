import numpy as np
import torch

from ml.models.model import ModelType, Model

if __name__ == '__main__':
    model = Model.get_model_class(ModelType.MNIST)()
    model.load_state_dict(torch.load('transfer_scripts/mnist-10-epoch.pth'))
    model.prepare_model_for_transfer_learning(26)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(sum([np.prod(p.size()) for p in model_parameters]))
