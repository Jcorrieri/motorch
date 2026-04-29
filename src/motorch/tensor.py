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
    "Register an __array_function__ implementation for MoTensor objects."
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator

# --- Supported Numpy Functions --- #

@implements(np.stack)
def tensor_stack(tup, **kwargs):
    return Tensor(np.stack([t.data for t in tup], **kwargs))

@implements(np.column_stack)
def tensor_colstack(tup):
    return Tensor(np.column_stack([t.data for t in tup]))

@implements(np.concatenate)
def tensor_concat(tup, **kwargs):
    return Tensor(np.concatenate([t.data for t in tup], **kwargs))

@implements(np.ones)
def tensor_ones(shape, **kwargs):
    return Tensor(np.ones(shape, **kwargs))

@implements(np.zeros)
def tensor_zeros(shape, **kwargs):
    return Tensor(np.zeros(shape, **kwargs))

@implements(np.ones_like)
def tensor_ones_like(arr, **kwargs):
    return Tensor(np.ones_like(arr.data, **kwargs))

@implements(np.zeros_like)
def tensor_zeros_like(arr, **kwargs):
    return Tensor(np.zeros_like(arr.data, **kwargs))

@implements(np.empty)
def tensor_empty(shape, **kwargs):
    return Tensor(np.empty(shape, **kwargs))

@implements(np.empty_like)
def tensor_empty_like(arr, **kwargs):
    return Tensor(np.empty_like(arr.data, **kwargs))

@implements(np.transpose)
def tensor_transpose(arr, axes=None):
    return Tensor(np.transpose(arr.data, axes))

@implements(np.reshape)
def tensor_reshape(arr, shape, **kwargs):
    return Tensor(np.reshape(arr.data, shape, **kwargs))

@implements(np.where)
def tensor_where(cond, x, y):
    return Tensor(np.where(cond.data, x, y))

@implements(np.clip)
def tensor_clip(a, *args, **kwargs):
    return Tensor(np.clip(a.data, *args, **kwargs))

@implements(np.exp)
def tensor_exp(arr, **kwargs):
    result = np.exp(arr.data, **kwargs)
    if np.ndim(result) == 0:
        return result.item()  # return plain scalar
    return Tensor(result)

@implements(np.log1p)
def tensor_log1p(arr, **kwargs):
    result = np.log1p(arr.data, **kwargs)
    if np.ndim(result) == 0:
        return result.item()  # return plain scalar
    return Tensor(result)

@implements(np.sum)
def tensor_sum(arr, **kwargs):
    result = np.sum(arr.data, **kwargs)
    if np.ndim(result) == 0:
        return result.item()  # return plain scalar
    return Tensor(result)

@implements(np.mean)
def tensor_mean(arr, **kwargs):
    result = np.mean(arr.data, **kwargs)
    if np.ndim(result) == 0:
        return result.item()  # return plain scalar
    return Tensor(result)


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

    def __iter__(self):
        for item in self.data:
            yield Tensor(item)

    def __getitem__(self, key):
        """Allows for array slicing"""
        return Tensor(self.data[key])

    def __setitem__(self, key, value):
        """Allows for setting elements via slicing"""
        if isinstance(value, Tensor):
           value = value.data 
        self.data[key] = value

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
        # Properly handle in-place operations (e.g. +=, -=)
        if 'out' in kwargs:
            kwargs['out'] = tuple(
                o.data if isinstance(o, Tensor) else o
                for o in kwargs['out']
            )
        result = getattr(ufunc, method)(*unwrapped, **kwargs)
        if isinstance(result, np.ndarray):
            return Tensor(result)
        return result  # scalars, None, etc.

    # Allows for overriding np.mean, np.sum, etc. as motorch.mean, motorch.sum
    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle Tensor objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    # --- Intrinsic Numpy Function Support (e.g. Tensor.mean(), Tensor.exp()) --- #

    def sum(self, **kwargs):
        "Implementation of np.sum for motorch.Tensor objects"
        return Tensor(np.sum(self, **kwargs))

    def mean(self, **kwargs):
        "Implementation of np.mean for motorch.Tensor objects"
        return Tensor(np.mean(self, **kwargs))

    def transpose(self, axes=None):
        """Returns a tensor with axes transposed."""
        return Tensor(np.transpose(self, axes))

    def reshape(self, *args, **kwargs):
        """Implementation of np.reshape for motorch.tensor objects."""
        return Tensor(np.reshape(self, args, **kwargs))

    def item(self, *args):
        return self.data.item(*args)

    def numpy(self) -> np.ndarray:
        return self.data

    # --- Properties --- #

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        return Tensor(self.data.T)


def tensor(data, dtype=None, requires_grad=False):
    """Factory function similar to torch.tensor()"""
    return Tensor(data, dtype=dtype, requires_grad=requires_grad)

