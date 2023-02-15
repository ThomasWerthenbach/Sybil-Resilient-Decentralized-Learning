import numpy as np

from ml.models.model import Model

if __name__ == '__main__':
    model = Model.get_model_class("MNIST")()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print(sum([np.prod(p.size()) for p in model_parameters]))
