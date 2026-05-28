"""Activation modules for MoTorch."""

from .module import Module
from motorch.utils import no_grad
from motorch.tensor import tensor
import motorch as mo
import motorch.nn.functional as F


class Sgn(Module):
    """Sign activation module that returns +1 for non-negative inputs."""

    def forward(self, x):
        return mo.where(x >= 0, 1, -1)


class Sigmoid(Module):
    """Sigmoid activation module with stored gradient support."""

    def forward(self, x):
        """Compute the sigmoid activation and cache its gradient."""
        with no_grad():
            out = F.sigmoid(x)
        self.z = out
        
        result = tensor(out.data)

        self._backwards(result, [x], [self._grad()])

        return result

    def _grad(self):
        """Return the gradient of the sigmoid activation for input ``z``."""
        with no_grad():
            out = F.sigmoid_grad(self.z, precomputed=True)
        return out
