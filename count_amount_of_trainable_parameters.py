import numpy as np
import torch

from ml.models.model import ModelType, Model

if __name__ == '__main__':
    model = Model.get_model_class(ModelType.MNIST)()
    model.load_state_dict(torch.load('transfer_scripts/98.96acc.mnist.pth'))
    model.prepare_for_transfer_learning(10)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(sum([np.prod(p.size()) for p in model_parameters]))
