"""Core Tensor implementation and NumPy interoperability layer.

The :mod:`motorch.tensor` module defines the :class:`Tensor` class and helper
functions that wrap NumPy operations while preserving the custom Tensor type.

References:
    - https://numpy.org/doc/stable/reference/generated/numpy.ndarray
    - https://numpy.org/doc/stable/user/basics.dispatch.html
"""

from types import FunctionType
from typing import Optional
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from motorch.utils import _requires_grad, resolve_local_grads
from motorch.utils.no_grad import no_grad


# --- __array_function__ implementations --- #

def tensor_stack(tup, **kwargs):
    return Tensor(np.stack([t.data for t in tup], **kwargs))

def tensor_colstack(tup):
    return Tensor(np.column_stack([t.data for t in tup]))

def tensor_concat(tup, **kwargs):
    return Tensor(np.concatenate([t.data for t in tup], **kwargs))

def tensor_ones_like(arr, **kwargs):
    return Tensor(np.ones_like(arr.data, **kwargs))

def tensor_zeros_like(arr, **kwargs):
    return Tensor(np.zeros_like(arr.data, **kwargs))

def tensor_empty_like(arr, **kwargs):
    return Tensor(np.empty_like(arr.data, **kwargs))

def tensor_transpose(arr, axes=None):
    return Tensor(np.transpose(arr.data, axes))

def tensor_reshape(arr, shape, **kwargs):
    return Tensor(np.reshape(arr.data, shape, **kwargs))

def tensor_where(cond, x, y):
    return Tensor(np.where(cond.data, x, y))

def tensor_clip(a, *args, **kwargs):
    return Tensor(np.clip(a.data, *args, **kwargs))

def tensor_exp(arr, **kwargs):
    result = np.exp(arr.data, **kwargs)
    return Tensor(result)

def tensor_log1p(arr, **kwargs):
    result = np.log1p(arr.data, **kwargs)
    return Tensor(result)

def tensor_sum(arr, **kwargs):
    result = np.sum(arr.data, **kwargs)
    return Tensor(result)

def tensor_mean(arr, **kwargs):
    result = np.mean(arr.data, **kwargs)
    return Tensor(result)

def tensor_sqrt(arr, **kwargs):
    result = np.sqrt(arr.data, **kwargs)
    return Tensor(result)


class Tensor(NDArrayOperatorsMixin):
    """A thin Tensor wrapper around a NumPy ndarray.

    This class stores the underlying array in ``self.data`` and exposes
    NumPy interoperability through ``__array__``, ``__array_function__`` and
    ``__array_ufunc__`` hooks. It also carries optional gradient tracking
    metadata for use in simple neural network computations.
    """

    def __init__(self, data, requires_grad=False, **kwargs) -> None:
        self.data = np.array(data, **kwargs) # always copies data
        self.grad: Optional[Tensor] = None
        self.grad_fn: Optional[FunctionType] = None
        self._backwards = lambda: None
        self._children = []
        self._version = 0
        self.requires_grad = requires_grad

    # --- __Methods__ --- #

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        return f"Tensor({self.data})"

    def __format__(self, format_spec):
        if (self.ndim > 1):
            return self.__repr__()
        return format(self.item(), format_spec)

    def __float__(self):
        return float(self.item())

    def __int__(self):
        return int(self.item())

    def __iter__(self):
        for item in self.data:
            yield Tensor(item)

    def __getitem__(self, key):
        """Return a sliced Tensor view of the underlying data."""
        return Tensor(self.data[key])

    def __setitem__(self, key, value):
        """Set array values using standard NumPy-style indexing."""
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
        def unwrap(x):
            if isinstance(x, Tensor):
                return x.data
            return x

        unwrapped = [unwrap(input) for input in inputs]

        # Properly handle in-place operations (e.g. +=, -=)
        if 'out' in kwargs:
            kwargs['out'] = tuple(
                o.data if isinstance(o, Tensor) else o
                for o in kwargs['out']
            )
        result = getattr(ufunc, method)(*unwrapped, **kwargs)
        if result is None:
            return None

        # Create graph
        def convert_to_tensor(x):
            if not isinstance(x, Tensor):
                return tensor(x, requires_grad=False)
            return x

        # ERROR: multiple instances of the same child will have grad_fn be overridden!
        inputs_t = [convert_to_tensor(input) for input in inputs]
        requires_grad = _requires_grad(inputs_t)
        result_t = tensor(result, requires_grad=requires_grad) 
        if requires_grad: 
            result_t._children = inputs_t
            local_grads = resolve_local_grads(ufunc, unwrapped)
            result.grad_fn = lambda _g=local_grads: self._grad_fn(result_t, inputs_t, _g)

        # In-place op (numpy already mutated self.data, return self)
        if 'out' in kwargs and kwargs['out'][0] is self.data:
            if requires_grad:
                self._version += 1 # if versions are not consistent throw error during backprop
            return self

        return result_t

    def _grad_fn(self, z, inputs, local_grads):
        expected_version = z._version
        with no_grad():
            for i, x in enumerate(inputs):
                if not x.grad:
                    x.grad = tensor_zeros_like(x)
                x.grad += z.grad * local_grads[i]
                if x._version != expected_version:
                    raise ValueError(f"{x} has been modified illegally.")

    # --- Intrinsic Numpy Function Support (e.g. Tensor.mean(), Tensor.exp()) --- #

    def sum(self, **kwargs):
        "Implementation of np.sum for motorch.Tensor objects"
        return tensor_sum(self, **kwargs)

    def mean(self, **kwargs):
        "Implementation of np.mean for motorch.Tensor objects"
        return tensor_mean(self, **kwargs)

    def exp(self, **kwargs):
        "Implementation of np.exp for motorch.Tensor objects"
        return tensor_exp(self, **kwargs)

    def log1p(self, **kwargs):
        "Implementation of np.log1p for motorch.Tensor objects"
        return tensor_log1p(self, **kwargs)

    def clip(self, *args, **kwargs):
        return tensor_clip(self, *args, **kwargs)

    def transpose(self, axes=None):
        """Returns a tensor with axes transposed."""
        return tensor_transpose(self, axes=axes)

    def reshape(self, *args, **kwargs):
        """Implementation of np.reshape for motorch.tensor objects."""
        if len(args) == 1: # Handle tuples
            args = args[0]
        return tensor_reshape(self, args, **kwargs)

    # --- Other Functions --- #

    def item(self, *args):
        return self.data.item(*args)

    def numpy(self) -> np.ndarray:
        return self.data

    def backwards(self, keep_graph=False):
        self.grad = tensor_ones_like(self.data)
        stack = [self]
        while stack:
            node = stack.pop(0)
            if node.grad_fn:
                node.grad_fn()
            print(f"Node: {node}, Grad: {node.grad}")
            stack.extend(node._children)
            if not keep_graph:
                node._children = [] # only store graph until backwards() is called

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


# --- Standalone Functions --- #

def tensor(data, dtype=None, requires_grad=False, **kwargs):
    """Create a new ``Tensor`` from array-like data.

    Parameters:
        data: Array-like input data.
        dtype: Optional NumPy dtype for the created array.
        requires_grad: Whether gradient tracking should be enabled.
    """
    return Tensor(data, dtype=dtype, requires_grad=requires_grad, **kwargs)

def tensor_ones(shape, **kwargs):
    """Create a ``Tensor`` filled with ones."""
    return Tensor(np.ones(shape, **kwargs))

def tensor_zeros(shape, **kwargs):
    """Create a ``Tensor`` filled with zeros."""
    return Tensor(np.zeros(shape, **kwargs))

def tensor_empty(shape, **kwargs):
    """Create an uninitialized ``Tensor`` of the given shape."""
    return Tensor(np.empty(shape, **kwargs))

