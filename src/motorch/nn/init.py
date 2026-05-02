"""
Weight Initialization Methods
Ref: https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py

Random Initialization relies on a passed generator object for now.
"""
import math
from motorch import Tensor, zeros_like


def from_tensor(parameter, value):
    if isinstance(value, Tensor):
        parameter.data = value.data 
    else:
        parameter.data = value

def glorot(rng, weight):
    """
    parameters:
        rng: A numpy random generator object
        weight: A weight tensor
    """
    fan_in, fan_out = weight.shape
    gl_x = math.sqrt(6 / (fan_in + fan_out))
    weight.data = rng.uniform(-gl_x, gl_x, (fan_in, fan_out))

def zeros(weight):
    zero_tensor = zeros_like(weight)
    from_tensor(weight, zero_tensor)

