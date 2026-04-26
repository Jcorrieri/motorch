import numpy as np


class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False) -> None:
        self.data = np.asarray(data, dtype=dtype)
        self.grad = None
        self.requires_grad = requires_grad

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        return f"MoTensor({self.data})"

    def __add__(self, other):
        """Element-wise Addition"""
        if isinstance(other, Tensor):
            res = self.data + other.data
        else:
            res = self.data + other

        return Tensor(res)

    # Needed for scalar + tensor (element-wise)
    def __radd__(self, other):
        return self.__add__(other)

    @property
    def shape(self):
        """Returns the shape of the tensor as a tuple."""
        return self.data.shape


def tensor(data, requires_grad=False):
    """Factory function similar to torch.tensor()"""
    return Tensor(data, requires_grad=requires_grad)

