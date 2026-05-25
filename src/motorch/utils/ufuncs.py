"""Ufunc related utilities."""

import numpy as np


def resolve_local_grads(ufunc, items):
    SUPPORTED_UFUNCS = {
        np.multiply: lambda inputs: (inputs[1], inputs[0]),
        np.add: lambda _: (1, 1),
    }

    local_grad_fn = SUPPORTED_UFUNCS.get(ufunc, lambda _: [0 for _ in range(len(items))]) 
    return local_grad_fn(items)
