from motorch.tensor import Tensor


class Optimizer:
    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        for param in self.parameters:
            if isinstance(param, Tensor):
                param.grad = 0.0 # type: ignore[AttributeAccessIssue]

    def step(self):
        raise NotImplementedError

