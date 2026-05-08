"""Activation modules for MoTorch."""

import motorch as mo
from .module import Module
import motorch.nn.functional as F


class Sgn(Module):
    """Sign activation module that returns +1 for non-negative inputs."""

    def forward(self, x):
        return mo.where(x >= 0, 1, -1)


class Sigmoid(Module):
    """Sigmoid activation module with stored gradient support."""

    def forward(self, x):
        """Compute the sigmoid activation and cache its gradient."""
        self.grad = self._grad(x)
        return F.sigmoid(x)

    def _grad(self, x):
        """Return the gradient of the sigmoid activation for input ``x``."""
        return F.sigmoid_grad(x)
