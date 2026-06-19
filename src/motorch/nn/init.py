"""
Weight Initialization Methods
Ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py

Random Initialization relies on a passed generator object for now.
"""

import math
from motorch import Tensor, zeros_like
from motorch.utils.no_grad import no_grad


def from_tensor(parameter, value):
    """Copy data from a tensor or raw array into a parameter."""
    if isinstance(value, Tensor):
        parameter.data = value.data 
    else:
        parameter.data = value

def glorot(rng, weight, gain=1.0):
    """Initialize weights using the Glorot/Xavier uniform scheme."""
    with no_grad():
        fan_in, fan_out = weight.shape
        gl_x = math.sqrt(6 / (fan_in + fan_out)) * gain
        weight.data = rng.uniform(-gl_x, gl_x, (fan_in, fan_out))

def zeros(weight):
    """Set a weight tensor to all zeros."""
    with no_grad():
        zero_tensor = zeros_like(weight)
        from_tensor(weight, zero_tensor)

