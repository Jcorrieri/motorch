"""
Core package for MoTorch modules, which is loosely based on the PyTorch package structure:
https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/modules
"""

from .module import Module
from .linear import Linear
from .loss import LogisticLoss
from .activations import (
    Sgn,
    Sigmoid,
    AltSigmoid
)

__all__ = [
    "Module",
    "Linear",
    "LogisticLoss",
    "Sgn",
    "Sigmoid",
    "AltSigmoid"
]
