import torch

class Scaler:
    def __init__(self, parameters, delta):
        self.parameters = parameters
        self.delta = delta

    def step(self):
        """Multiplies gradients in place."""
        for param in self.parameters:
            if param.grad is None:
                raise ValueError("backward() has to be called before running scaler")

            param.grad *= self.delta

if __name__ == '__main__':
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
    )

    scaler = Scaler(model.parameters(), delta=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    X, y = torch.randn(64, 10), torch.randn(64)

    # Optimization loop
    EPOCHS = 10
    for _ in range(EPOCHS):
        print("===============================")
        print("===============================")
        print("===============================")
        print("===============================")
        print(next(model.parameters()))
        output = model(X)
        loss = criterion(output, y)

        loss.backward()  # Now model has the gradients

        optimizer.step()  # Optimize model's parameters

        print(next(model.parameters()).grad)
        print(sum(next(model.parameters()).grad))

        scaler.step()  # Scaler gradients

        optimizer.zero_grad()  # Zero gradient before next step
        print(next(model.parameters()))