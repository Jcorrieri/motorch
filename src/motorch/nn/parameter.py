"""
Ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/parameter.py
This is mainly a wrapper class to distinct trainable and non-trainable tensors.
"""
from motorch import Tensor


class Parameter(Tensor):
    def __init__(self, data, requires_grad=True, **kwargs):
        """Since we do not have underlying C logic, we can use __init__ directly."""
        super().__init__(data, requires_grad, **kwargs)

    def __repr__(self):
        return f"Parameter({self.data})"
