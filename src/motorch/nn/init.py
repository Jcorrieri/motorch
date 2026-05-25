"""
Weight Initialization Methods
Ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py

Random Initialization relies on a passed generator object for now.
"""

import math
from motorch import Tensor, zeros_like


def from_tensor(parameter, value):
    """Copy data from a tensor or raw array into a parameter."""
    if isinstance(value, Tensor):
        parameter.data = value.data 
    else:
        parameter.data = value

def glorot(rng, weight):
    """Initialize weights using the Glorot/Xavier uniform scheme."""
    fan_in, fan_out = weight.shape
    gl_x = math.sqrt(6 / (fan_in + fan_out))
    weight.data = rng.uniform(-gl_x, gl_x, (fan_in, fan_out))

def zeros(weight):
    """Set a weight tensor to all zeros."""
    zero_tensor = zeros_like(weight)
    from_tensor(weight, zero_tensor)

