"""
Functional components for MoTorch. PyTorch delegates these to underlying C logic, but for simplicity the
explicit implementations are defined here, in Python.
"""

import motorch as mo
from motorch import Tensor


# --- Activations --- #


def sigmoid(x):
    """Compute the sigmoid activation for a tensor input."""
    x_clipped = mo.clip(x, -700, 700)  # to avoid numerical instability
    return 1 / (1 + mo.exp(-x_clipped))


def sigmoid_grad(x, precomputed=False):
    """
    Compute the gradient of the sigmoid activation.

    args:
        - x (Tensor): the input data
        - precomputed (boolean): whether x is the output of sigmoid
    """
    if not precomputed:
        x = sigmoid(x)
    return x * (1 - x)


# --- Loss Functions --- #  TODO: Change name to explicit +1/-1 loss.


def logloss(logits, labels):
    """Compute the average logistic loss for binary labels in {+1, -1}."""
    return mo.mean(mo.log1p(mo.exp(-logits * labels)))


def logloss_grad(logits, labels):
    """Return the gradient of the logistic loss w.r.t. logits."""
    return mo.tensor(-labels * sigmoid(-labels * logits) / len(labels))


# --- Layers --- #


def linear(x, weight, bias):
    """Compute a linear transformation with optional bias.

    Parameters:
        x: Input tensor with last dimension matching ``weight.shape[0]``.
        weight: Weight tensor of shape ``(in_features, out_features)``.
        bias: Bias tensor added to each row of the result.
    """
    if not isinstance(x, Tensor):
        raise ValueError("Input must be of type motorch.Tensor")

    assert x.shape[-1] == weight.shape[0], (
        f"Expected x.shape = ({len(x)}, {weight.shape[0]}), got {x.shape}."
    )

    return x @ weight + bias
