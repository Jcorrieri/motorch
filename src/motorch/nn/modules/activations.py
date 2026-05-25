"""Activation modules for MoTorch."""

from .module import Module
from motorch.utils import _track_grads, no_grad
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
        self.z = out

        track_grads = _track_grads([x])
        result = tensor(out.data, requires_grad=track_grads)
        if track_grads:
            result._children = [x] # store entire nd tensor as single child
            local_grad = self._grad()
            x.grad_fn = lambda _g=local_grad: result.grad * _g
            result.grad = tensor_zeros_like(result.data).numpy()

        return result

    def _grad(self):
        """Return the gradient of the sigmoid activation for input ``z``."""
        with no_grad():
            out = F.sigmoid_grad(self.z, precomputed=True)
        return out
