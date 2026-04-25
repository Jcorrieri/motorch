"""
Package for motorch.nn which loosely follows the PyTorch.nn structure:
https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn
"""

from . import functional
from .modules import (
    Linear,
    Model,
    LogisticLoss,
    Sgn,
    AltSigmoid,
    Sigmoid,
)

__all__ = [
    "Linear",
    "Model",
    "LogisticLoss",
    "Sgn",
    "Sigmoid",
    "AltSigmoid",
    "functional"
]
