"""Base optimizer abstractions for MoTorch."""

from motorch.tensor import Tensor, tensor_zeros_like
from motorch.utils.no_grad import no_grad


class Optimizer:
    """Base class for parameter optimizers."""

    def __init__(self, parameters):
        self.parameters = list(parameters) # convert to list for reuse (generators are one-time only).

    def zero_grad(self):
        """Reset gradients to zero for all tracked parameters."""
        for param in self.parameters:
            if isinstance(param, Tensor):
                param.grad = tensor_zeros_like(param)

    def step(self):
        """Apply a single optimization step.

        Subclasses should override _step with the specific update rule.
        """
        with no_grad():
            self._step()

    def _step(self):
        raise NotImplementedError

