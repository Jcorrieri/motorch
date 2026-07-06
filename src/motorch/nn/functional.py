"""
Functional components for MoTorch. PyTorch delegates these to underlying C logic, but for simplicity the
explicit implementations are defined here, in Python.
"""

import motorch as mo
from motorch import Tensor


# --- Activations --- #


def sigmoid(z):
    """Compute the sigmoid activation for a tensor input."""
    z_clipped = mo.clip(z, -700, 700)  # to avoid numerical instability
    return 1 / (1 + mo.exp(-z_clipped))


def sigmoid_grad(z, precomputed=False):
    """
    Compute the gradient of the sigmoid activation.

    args:
        - z (Tensor): the input data
        - precomputed (boolean): whether z is the output of sigmoid
    """
    if not precomputed:
        z = sigmoid(z)
    return z * (1 - z)


def relu(z):
    return mo.where(z > 0, z, 0)


def relu_grad(z):
    return mo.where(z > 0, 1, 0)


# --- Loss Functions --- #  TODO: Change name to explicit +1/-1 loss.


def logloss(logits, labels):
    """Compute the average logistic loss for binary labels in {+1, -1}."""
    return mo.mean(mo.log1p(mo.exp(-logits * labels)))


def logloss_grad(logits, labels):
    """Return the gradient of the logistic loss w.r.t. logits."""
    return mo.tensor(-labels * sigmoid(-labels * logits) / len(labels))


# --- Layers --- #


def linear(z, weight, bias):
    """Compute a linear transformation with optional bias.

    Parameters:
        z: Input tensor with last dimension matching ``weight.shape[0]``.
        weight: Weight tensor of shape ``(in_features, out_features)``.
        bias: Bias tensor added to each row of the result.
    """
    if not isinstance(z, Tensor):
        raise ValueError("Input must be of type motorch.Tensor")

    assert z.shape[-1] == weight.shape[0], (
        f"Expected z.shape = ({len(z)}, {weight.shape[0]}), got {z.shape}."
    )

    return z @ weight + bias
