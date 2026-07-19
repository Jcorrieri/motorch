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


def softmax(z):
    normalized = z - mo.max(z, axis=-1, keepdims=True)
    numerator = mo.exp(normalized)
    denominator = mo.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator


# --- Loss Functions --- #  TODO: Change name to explicit +1/-1 loss.


def logloss(logits, labels):
    """Compute the average logistic loss for binary labels in {+1, -1}."""
    return mo.mean(mo.log1p(mo.exp(-logits * labels)))


def logloss_grad(logits, labels):
    """Return the gradient of the logistic loss w.r.t. logits."""
    return mo.tensor(-labels * sigmoid(-labels * logits) / len(labels))


def cross_entropy(logits, labels):
    """
    It is numerically unstable to compute e^{zi} / Σ_{j}(e^{zj}) directly, since a large/small z would cause numerical overflow/underflow.
    plus, if all logits on the bottom are zero due to small z, you get a divide by zero error!

    Derivation of cross_entropy (NLLLoss)

        H(p, q) = -Σplog(q) over all classes c ∈ C
    
        When p = [0, 0, ..., 1, ..., 0], H(p, q) reduces to:
            -p_{c}log(q_{c}), where p_{c} = 1, p_{~c} = 0
            ⇒ -(1.0)log(Softmax(q_{c}))
            ⇒ -log(e^{qc} / Σ_{j}(e^{qj}))
            ⇒ -log(e^{qc}) + log(Σ_{j}(e^{qj}))
            ⇒ -qc + log(sum(e^{qj}))

    Next, employ the logsumexp trick to stabilize the right term
        - If sum(e^{...}) = 0, then we will have log(0) which is undefined
        - Large z will lead to undefined behavior (although mo.exp() is clamped to [-800, 800])
    
    LogSumExp invovles subtracting each logit by the maximum logit to avoid tricky exponentiation.
    
        log(sum(e^{qj}))
        ⇒ log(sum(e^{qj} * e^{m} * e^{-m})) -- (here, introduce m via e^{m} * e^{-m} = 1)
        ⇒ log(e^{m}sum(e^{qj - m})) -- (move e^{m} out since its constant)
        ⇒ m + log(sum(e^{qj - m}))

    This implementation currently only supports one-hot labels.
    """
    target_idx = mo.argmax(labels, axis=-1, keepdims=True)
    logit_term = mo.take_along_axis(logits, target_idx, axis=-1)
    m = mo.max(logits, axis=-1, keepdims=True)
    log_sum_exp = m + mo.log(mo.sum(mo.exp(logits - m), axis=-1, keepdims=True))
    return mo.mean(-logit_term + log_sum_exp)


def cross_entropy_grad(logits, labels):
    batch_size = 1 if logits.ndim == 1 else len(labels)
    return (softmax(logits) - labels) / batch_size


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
