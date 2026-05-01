import motorch as mo
from .module import Module
import motorch.nn.functional as F


class Sgn(Module):
    def forward(self, x):
        return mo.where(x >= 0, 1, -1)


class Sigmoid(Module):
    def forward(self, x):
        self.grad = self._grad(x)
        return F.sigmoid(x)

    def _grad(self, x):
        return F.sigmoid_grad(x)
