"""Tests for MoTorch loss modules."""

import numpy as np

import motorch.nn as nn
from motorch.tensor import Tensor, tensor


class TestCrossEntropyLoss:
    def test_forward_returns_expected_scalar_tensor(self):
        loss_function = nn.CrossEntropyLoss()
        logits = tensor([1.0, 2.0, 3.0])
        labels = tensor([0.0, 0.0, 1.0])
        expected = -3.0 + np.log(np.exp([1.0, 2.0, 3.0]).sum())

        result = loss_function(logits, labels)

        assert isinstance(result, Tensor)
        np.testing.assert_allclose(result.data, expected)

    def test_backward_populates_logits_gradient(self):
        loss_function = nn.CrossEntropyLoss()
        logits = tensor([1.0, 2.0, 3.0], requires_grad=True)
        labels = tensor([0.0, 0.0, 1.0])
        probabilities = np.exp(logits.data)
        probabilities /= probabilities.sum()

        loss_function(logits, labels).backward()

        np.testing.assert_allclose(logits.grad.data, probabilities - labels.data)

    def test_2d_backward_matches_mean_reduction(self):
        loss_function = nn.CrossEntropyLoss()
        logits = tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]], requires_grad=True)
        labels = tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        normalized_logits = logits.data - logits.data.max(axis=-1, keepdims=True)
        probabilities = np.exp(normalized_logits)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)

        loss = loss_function(logits, labels)
        loss.backward()

        assert loss.shape == ()
        expected = (probabilities - labels.data) / len(logits)
        np.testing.assert_allclose(logits.grad.data, expected)
