"""
Package for motorch.nn which loosely follows the PyTorch.nn structure:
https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn
"""

from . import functional
from . import init
from .parameter import Parameter
from .modules import (
    Linear,
    Module,
    LogisticLoss,
    Sgn,
    Sigmoid,
    AltSigmoid,
    ReLU,
    CrossEntropyLoss,
)

__all__ = [
    "Linear",
    "Module",
    "LogisticLoss",
    "CrossEntropyLoss",
    "Sgn",
    "ReLU",
    "Sigmoid",
    "AltSigmoid",
    "functional",
    "init",
    "Parameter",
]
