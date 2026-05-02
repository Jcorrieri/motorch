"""
MoTorch - a torch-like deep learning library.
"""

from .tensor import (
    Tensor,
    tensor,
    tensor_colstack as column_stack,
    tensor_stack as stack,
    tensor_concat as concatenate,
    tensor_ones as ones,
    tensor_ones_like as ones_like,
    tensor_zeros as zeros,
    tensor_zeros_like as zeros_like,
    tensor_empty as empty,
    tensor_empty_like as empty_like,
    tensor_mean as mean,
    tensor_exp as exp,
    tensor_sum as sum,
    tensor_log1p as log1p,
    tensor_sqrt as sqrt,
    tensor_transpose as transpose,
    tensor_reshape as reshape,
    tensor_where as where,
    tensor_clip as clip
)

__all__ = [
    "Tensor",
    "tensor",
    "column_stack",
    "stack",
    "concatenate",
    "ones",
    "ones_like",
    "zeros",
    "zeros_like",
    "empty",
    "empty_like",
    "mean",
    "exp",
    "sum",
    "log1p",
    "sqrt",
    "transpose",
    "reshape",
    "where",
    "clip"
]
