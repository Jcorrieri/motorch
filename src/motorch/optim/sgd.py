"""Stochastic gradient descent optimizer."""

from motorch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """Simple stochastic gradient descent optimizer."""

    def __init__(self, parameters, lr=0.001) -> None:
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        """Update each parameter using its stored gradient and learning rate."""
        for param in self.parameters: 
            param -= self.lr * param.grad

