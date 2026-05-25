"""Ufunc related utilities."""
import numpy as np


def resolve_local_grads(ufunc, items):
    SUPPORTED_UFUNCS = {
        np.multiply: (items[1], items[0]),
        np.add: (1, 1),
    }

    return SUPPORTED_UFUNCS[ufunc]
