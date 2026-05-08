"""Base optimizer abstractions for MoTorch."""

from motorch.tensor import Tensor


class Optimizer:
    """Base class for parameter optimizers."""

    def __init__(self, parameters):
        self.parameters = list(parameters) # convert to list for reuse (generators are one-time only).

    def zero_grad(self):
        """Reset gradients to zero for all tracked parameters."""
        for param in self.parameters:
            if isinstance(param, Tensor):
                param.grad = 0.0 # type: ignore[AttributeAccessIssue]

    def step(self):
        """Apply a single optimization step.

        Subclasses should override this method with the specific update rule.
        """
        raise NotImplementedError

