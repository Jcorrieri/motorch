"""
The MoTorch Tensor class is a custom ndarray container.

References:
    - https://numpy.org/doc/stable/reference/generated/numpy.ndarray
    - https://numpy.org/doc/stable/user/basics.dispatch.html
"""

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin


HANDLED_FUNCTIONS = {}

def implements(np_function):
    "Register an __array_function__ implementation for DiagonalArray objects."
    def decorator(func):
       HANDLED_FUNCTIONS[np_function] = func
       return func
    return decorator

def reduction(np_func):
    def method(self, **kwargs):
        # out covers np.mean(.., out=...)
        if "out" in kwargs and isinstance(kwargs["out"], Tensor):
            kwargs["out"] = kwargs["out"].data
        result = np_func(self.data, **kwargs)
        if isinstance(result, np.ndarray):
            return tensor(result)
        return result  # scalars when no axis specified
    return method


class Tensor(NDArrayOperatorsMixin):
    def __init__(self, data, requires_grad=False, **kwargs) -> None:
        self.data = np.array(data, **kwargs) # always copies data
        self.grad = None
        self.requires_grad = requires_grad

    # --- __Methods__ --- #

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        return f"MoTensor({self.data})"

    # for np.asarray(...)
    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError(
                "`copy=False` isn't supported. A copy is always created."
            )
        if dtype:
            return self.data.astype(dtype)
        else:
            return self.data.copy()

    # handles __add__, __mul__, etc.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        unwrapped = [
            input.data if isinstance(input, Tensor) else input
            for input in inputs
        ]
        result = getattr(ufunc, method)(*unwrapped, **kwargs)
        if isinstance(result, np.ndarray):
            return tensor(result)
        return result  # scalars, None, etc.

    # handles mean, sum, etc.
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    # --- NumPy Function Support --- #

    @implements(np.sum)
    def tensor_sum(arr, **kwargs):
        "Implementation of np.sum for motorch.Tensor objects"
        return np.sum(arr.data, **kwargs)

    @implements(np.mean)
    def tensor_mean(arr, **kwargs):
        "Implementation of np.mean for motorch.Tensor objects"
        return np.mean(arr.data, **kwargs)

    @implements(np.transpose)
    def tensor_transpose(arr, axes=None):
        """Returns a tensor with axes transposed."""
        return Tensor(np.transpose(arr.data, axes))

    @implements(np.reshape)
    def tensor_reshape(arr, **kwargs):
        """Implementation of np.reshape for motorch.tensor objects."""
        return Tensor(np.reshape(arr.data, **kwargs))

    def T(self):
        """Implementation of ndarry.T for motorch tensors."""
        return Tensor(self.data.T)

    # --- Intrinsic Functions --- #

    mean = reduction(np.mean)
    sum  = reduction(np.sum)
    std  = reduction(np.std)
    min  = reduction(np.min)
    max  = reduction(np.max)
    transpose = reduction(np.transpose)
    reshape = reduction(np.reshape)

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

