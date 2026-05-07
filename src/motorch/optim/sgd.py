from motorch.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001) -> None:
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters: 
            param -= self.lr * param.grad

