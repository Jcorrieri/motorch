"""Activation modules for MoTorch."""

from motorch.autograd.forward import apply_forward_pass
from .module import Module
from motorch.utils import no_grad
from motorch.tensor import tensor
import motorch as mo
import motorch.nn.functional as F


class Sgn(Module):
    """Sign activation module that returns +1 for non-negative inputs."""

    def forward(self, z):
        return mo.where(z >= 0, 1, -1)


class Sigmoid(Module):
    """Sigmoid activation module with stored gradient support."""

    def forward(self, z):
        """Compute the sigmoid activation and cache its gradient."""
        with no_grad():
            out = F.sigmoid(z)

        result = tensor(out.data)
        local_grad = [self._grad_fn(result)]
        apply_forward_pass(result, [z], local_grad)

        return result

    def _grad_fn(self, z):
        """Compute the gradient of the sigmoid activation w.r.t the input z."""
        with no_grad():
            out = F.sigmoid_grad(z, precomputed=True)
        return out


class AltSigmoid(Module):
    """Sigmoid function rescaled to output values between -1 and 1."""

    def forward(self, z):
        with no_grad():
            out = 2 * F.sigmoid(z) - 1

        result = tensor(out.data)
        local_grad = [self._grad_fn(z)]
        apply_forward_pass(result, [z], local_grad)

        return result

    def _grad_fn(self, z):
        with no_grad():
            out = 2 * F.sigmoid_grad(z)
        return out


class ReLU(Module):
    """ReLU activation function."""

    def forward(self, z):
        with no_grad():
            out = F.relu(z)

        result = tensor(out.data)
        local_grad = [self._grad_fn(result)]
        apply_forward_pass(result, [z], local_grad)

        return result

    def _grad_fn(self, z):
        with no_grad():
            out = F.relu_grad(z)
        return out

