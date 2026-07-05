"""Ufunc related utilities."""

import numpy as np

from motorch.utils.no_grad import no_grad


def resolve_local_grads(ufunc, items):
    with no_grad():
        SUPPORTED_UFUNCS = {
            # Basic arithmetic
            np.multiply: lambda inputs: (inputs[1], inputs[0]),
            np.add: lambda _: (1, 1),
            np.subtract: lambda _: (1, -1),
            np.true_divide: lambda inputs: (1 / inputs[1], -inputs[0] / inputs[1] ** 2),
            np.divide: lambda inputs: (1 / inputs[1], -inputs[0] / inputs[1] ** 2),
            np.negative: lambda _: (-1,),
            np.power: lambda inputs: (
                inputs[1] * inputs[0] ** (inputs[1] - 1),
                inputs[0] ** inputs[1] * np.log(inputs[0]),
            ),
            # Exponential / logarithmic
            np.exp: lambda inputs: (np.exp(inputs[0]),),
            np.exp2: lambda inputs: (np.exp2(inputs[0]) * np.log(2),),
            np.log: lambda inputs: (1 / inputs[0],),
            np.log2: lambda inputs: (1 / (inputs[0] * np.log(2)),),
            np.log10: lambda inputs: (1 / (inputs[0] * np.log(10)),),
            np.log1p: lambda inputs: (1 / (1 + inputs[0]),),
            np.sqrt: lambda inputs: (1 / (2 * np.sqrt(inputs[0])),),
            np.square: lambda inputs: (2 * inputs[0],),
            np.reciprocal: lambda inputs: (-1 / inputs[0] ** 2,),
            np.abs: lambda inputs: (np.sign(inputs[0]),),
            # Trigonometric
            np.sin: lambda inputs: (np.cos(inputs[0]),),
            np.cos: lambda inputs: (-np.sin(inputs[0]),),
            np.tan: lambda inputs: (1 / np.cos(inputs[0]) ** 2,),
            np.arcsin: lambda inputs: (1 / np.sqrt(1 - inputs[0] ** 2),),
            np.arccos: lambda inputs: (-1 / np.sqrt(1 - inputs[0] ** 2),),
            np.arctan: lambda inputs: (1 / (1 + inputs[0] ** 2),),
            # Hyperbolic
            np.sinh: lambda inputs: (np.cosh(inputs[0]),),
            np.cosh: lambda inputs: (np.sinh(inputs[0]),),
            np.tanh: lambda inputs: (1 - np.tanh(inputs[0]) ** 2,),
        }

        local_grad_fn = SUPPORTED_UFUNCS.get(
            ufunc, lambda _: [0 for _ in range(len(items))]
        )
        return local_grad_fn(items)
