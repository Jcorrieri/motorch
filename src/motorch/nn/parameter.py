"""
Ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/parameter.py
"""
import motorch as mo


class Parameter(mo.Tensor):
    def __init__(self, data, requires_grad=True, **kwargs):
        """Since we do not have underlying C logic, we can use __init__ directly."""
        super().__init__(data, requires_grad, **kwargs)

    def __repr__(self):
        return f"Parameter({self.data})"
