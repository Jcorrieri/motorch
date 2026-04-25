import numpy as np

from motorch.nn.functional import sigmoid, sigmoid_derivative


class Sgn():
    def __call__(self, x):
        return np.where(x >= 0, 1, -1)

    def __str__(self):
        return "Vector Sign Function: 1 if x >= 0, else -1\n"


class Sigmoid():
    def __call__(self, x):
        self.grad = self._grad(x)
        return sigmoid(x)

    def __str__(self) -> str:
        return "Sigmoid Function"

    def _grad(self, x):
        return sigmoid_derivative(x)


class AltSigmoid():
    def __call__(self, x):
        self.grad = self._grad(x)
        return 2 * sigmoid(x) - 1

    def __str__(self):
        return "Alternate Sigmoid: 2 * sigmoid(x) - 1\n"

    def _grad(self, x):
        return 2 * sigmoid_derivative(x)
