import numpy as np


def _binary_op(op):
    def method(self, other):
        if isinstance(other, Tensor):
            return Tensor(op(self.data, other.data))
        else:
            return Tensor(op(self.data, other))
    return method


class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False) -> None:
        self.data = np.asarray(data, dtype=dtype)
        self.grad = None
        self.requires_grad = requires_grad

    # --- Define Binary Operations --- #

    __add__  = _binary_op(np.add)
    __radd__ = _binary_op(np.add)
    __sub__  = _binary_op(np.subtract)
    __rsub__ = _binary_op(np.subtract)
    __mul__  = _binary_op(np.multiply)
    __rmul__ = _binary_op(np.multiply)
    __truediv__ = _binary_op(np.divide)
    __matmul__  = _binary_op(np.matmul)

    # --- Methods --- #

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        return f"MoTensor({self.data})"

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError(
                "`copy=False` isn't supported. A copy is always created."
            )
        if dtype:
            return self.data.astype(dtype)
        else:
            return self.data.copy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        pass

    # --- Properties --- #

    @property
    def shape(self):
        """Returns the shape of the tensor as a tuple."""
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim


def tensor(data, dtype=None, requires_grad=False):
    """Factory function similar to torch.tensor()"""
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)

