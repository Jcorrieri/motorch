"""
Ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/parameter.py
This is mainly a wrapper class to distinct trainable and non-trainable tensors.
"""
from motorch import Tensor


class Parameter(Tensor):
    """A tensor subclass that represents a learnable model parameter."""

    def __init__(self, data, requires_grad=True, **kwargs):
        """Initialize a :class:`Parameter` with optional gradient tracking."""
        super().__init__(data, requires_grad, **kwargs)

    def __repr__(self):
        return f"Parameter({self.data})"
