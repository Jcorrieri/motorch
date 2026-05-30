"""Activation modules for MoTorch."""

from .module import Module
from motorch.utils import no_grad, _requires_grad
from motorch.tensor import tensor, tensor_zeros_like
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

        requires_grad = _requires_grad([x])
        result = tensor(out.data, requires_grad=requires_grad)
        if requires_grad:
            result.grad_fn = lambda _z=result: self._grad_fn(_z, x)
            result._children = [x]

        return result

    def _grad_fn(self, z, x):
        """Compute the gradient of the sigmoid activation for input ``z``."""
        with no_grad():
            local_grad = F.sigmoid_grad(z, precomputed=True)
            if not x.grad:
                x.grad = tensor_zeros_like(x)
            x.grad += z.grad * local_grad

