"""Tests for motorch functional components."""

import numpy as np
import pytest
from motorch.tensor import tensor, Tensor
import motorch.nn.functional as F


# ── helpers ───────────────────────────────────────────────────────────────────


def assert_close(actual, expected, rtol=1e-5, atol=1e-7):
    np.testing.assert_allclose(
        actual.data if isinstance(actual, Tensor) else np.array(actual),
        expected if isinstance(expected, np.ndarray) else np.array(expected),
        rtol=rtol,
        atol=atol,
    )


# ── sigmoid ───────────────────────────────────────────────────────────────────


class TestSigmoid:
    def test_zero_input(self):
        # sigmoid(0) = 0.5
        assert_close(F.sigmoid(tensor(0.0)), 0.5)

    def test_large_positive(self):
        # sigmoid(large) -> 1.0
        assert_close(F.sigmoid(tensor(100.0)), 1.0, atol=1e-5)

    def test_large_negative(self):
        # sigmoid(-large) -> 0.0
        assert_close(F.sigmoid(tensor(-100.0)), 0.0, atol=1e-5)

    def test_known_value(self):
        # sigmoid(1) = 1 / (1 + e^-1)
        expected = 1 / (1 + np.exp(-1.0))
        assert_close(F.sigmoid(tensor(1.0)), expected)

    def test_output_range(self):
        x = tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        out = F.sigmoid(x)
        assert np.all(out.data > 0.0)
        assert np.all(out.data < 1.0)

    def test_vector_input(self):
        x = tensor([0.0, 1.0, -1.0])
        expected = 1 / (1 + np.exp(-np.array([0.0, 1.0, -1.0])))
        assert_close(F.sigmoid(x), expected)

    def test_numerical_stability_large_positive(self):
        # Should not overflow or return nan
        out = F.sigmoid(tensor(800.0))
        assert np.isfinite(out.data)

    def test_numerical_stability_large_negative(self):
        out = F.sigmoid(tensor(-800.0))
        assert np.isfinite(out.data)

    def test_returns_tensor(self):
        assert isinstance(F.sigmoid(tensor(1.0)), Tensor)


# ── sigmoid_grad ──────────────────────────────────────────────────────────────


class TestSigmoidGrad:
    def test_zero_input(self):
        # sigmoid_grad(0) = 0.5 * 0.5 = 0.25
        assert_close(F.sigmoid_grad(tensor(0.0)), 0.25)

    def test_known_value(self):
        s = 1 / (1 + np.exp(-1.0))
        expected = s * (1 - s)
        assert_close(F.sigmoid_grad(tensor(1.0)), expected)

    def test_precomputed_false(self):
        # Without precomputed, input is raw x
        x = tensor(1.0)
        result = F.sigmoid_grad(x, precomputed=False)
        s = 1 / (1 + np.exp(-1.0))
        assert_close(result, s * (1 - s))

    def test_precomputed_true(self):
        # With precomputed, input is already sigmoid(x)
        s = tensor(1 / (1 + np.exp(-1.0)))
        result = F.sigmoid_grad(s, precomputed=True)
        expected = float(s.data) * (1 - float(s.data))
        assert_close(result, expected)

    def test_precomputed_false_matches_precomputed_true(self):
        # Both paths should produce the same result
        x = tensor(2.0)
        s = F.sigmoid(x)
        grad_raw = F.sigmoid_grad(x, precomputed=False)
        grad_precomputed = F.sigmoid_grad(s, precomputed=True)
        assert_close(grad_raw, grad_precomputed.data)

    def test_output_range(self):
        # sigmoid_grad is always in (0, 0.25]
        x = tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
        out = F.sigmoid_grad(x)
        assert np.all(out.data > 0.0)
        assert np.all(out.data <= 0.25 + 1e-8)

    def test_symmetry(self):
        # sigmoid_grad is symmetric: grad(x) == grad(-x)
        x = tensor(1.5)
        assert_close(F.sigmoid_grad(x), F.sigmoid_grad(tensor(-1.5)).data)

    def test_returns_tensor(self):
        assert isinstance(F.sigmoid_grad(tensor(0.0)), Tensor)


# ── relu ──────────────────────────────────────────────────────────────────────


class TestRelu:
    def test_zero_input(self):
        assert_close(F.relu(tensor(0.0)), 0.0)

    def test_positive_input(self):
        assert_close(F.relu(tensor(2.5)), 2.5)

    def test_negative_input(self):
        assert_close(F.relu(tensor(-2.5)), 0.0)

    def test_vector_input(self):
        out = F.relu(tensor([-2.0, -0.5, 0.0, 0.5, 2.0]))
        assert_close(out, [0.0, 0.0, 0.0, 0.5, 2.0])

    def test_matrix_input(self):
        out = F.relu(tensor([[-1.0, 2.0], [0.0, -3.0]]))
        assert_close(out, [[0.0, 2.0], [0.0, 0.0]])

    def test_returns_tensor(self):
        assert isinstance(F.relu(tensor(1.0)), Tensor)


# ── relu_grad ─────────────────────────────────────────────────────────────────


class TestReluGrad:
    def test_zero_input(self):
        assert_close(F.relu_grad(tensor(0.0)), 0.0)

    def test_positive_input(self):
        assert_close(F.relu_grad(tensor(2.5)), 1.0)

    def test_negative_input(self):
        assert_close(F.relu_grad(tensor(-2.5)), 0.0)

    def test_vector_input(self):
        out = F.relu_grad(tensor([-2.0, -0.5, 0.0, 0.5, 2.0]))
        assert_close(out, [0.0, 0.0, 0.0, 1.0, 1.0])

    def test_returns_tensor(self):
        assert isinstance(F.relu_grad(tensor(1.0)), Tensor)


# ── logloss ───────────────────────────────────────────────────────────────────


class TestLogloss:
    def test_perfect_positive_prediction(self):
        # large positive logit, label +1 → loss near 0
        logits = tensor([100.0])
        labels = tensor([1.0])
        out = F.logloss(logits, labels)
        assert float(out.data) == pytest.approx(0.0, abs=1e-5)

    def test_perfect_negative_prediction(self):
        # large negative logit, label -1 → loss near 0
        logits = tensor([-100.0])
        labels = tensor([-1.0])
        out = F.logloss(logits, labels)
        assert float(out.data) == pytest.approx(0.0, abs=1e-5)

    def test_wrong_prediction_incurs_loss(self):
        # positive logit, negative label → high loss
        logits = tensor([10.0])
        labels = tensor([-1.0])
        out = F.logloss(logits, labels)
        assert float(out.data) > 5.0

    def test_zero_logit_known_value(self):
        # loss = log(1 + exp(0)) = log(2) ≈ 0.6931
        logits = tensor([0.0])
        labels = tensor([1.0])
        out = F.logloss(logits, labels)
        assert float(out.data) == pytest.approx(np.log(2), rel=1e-5)

    def test_mean_over_batch(self):
        # Two samples with known values
        logits = tensor([1.0, -1.0])
        labels = tensor([1.0, -1.0])
        # Each: log(1 + exp(-1))
        expected = np.log(1 + np.exp(-1.0))
        out = F.logloss(logits, labels)
        assert float(out.data) == pytest.approx(expected, rel=1e-5)

    def test_loss_non_negative(self):
        logits = tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        labels = tensor([1.0, -1.0, 1.0, 1.0, -1.0])
        out = F.logloss(logits, labels)
        assert float(out.data) >= 0.0

    def test_returns_tensor(self):
        assert isinstance(F.logloss(tensor([1.0]), tensor([1.0])), Tensor)


# ── logloss_grad ──────────────────────────────────────────────────────────────


class TestLoglossGrad:
    def test_correct_prediction_small_grad(self):
        # Large positive logit, label +1 → gradient near 0
        logits = tensor([100.0])
        labels = tensor([1.0])
        grad = F.logloss_grad(logits, labels)
        assert abs(float(grad.item())) < 1e-4

    def test_wrong_prediction_large_grad(self):
        # Large positive logit, label -1 → gradient near -1/n
        logits = tensor([100.0])
        labels = tensor([-1.0])
        grad = F.logloss_grad(logits, labels)
        assert abs(float(grad.item())) == pytest.approx(1.0, abs=1e-4)

    def test_known_value(self):
        # grad = -y * sigmoid(-y*x) / n
        # x=0, y=1, n=1: -1 * sigmoid(0) / 1 = -0.5
        logits = tensor([0.0])
        labels = tensor([1.0])
        grad = F.logloss_grad(logits, labels)
        assert float(grad.item()) == pytest.approx(-0.5, rel=1e-5)

    def test_scaled_by_batch_size(self):
        # Same single sample repeated twice — grad should halve
        logits1 = tensor([1.0])
        labels1 = tensor([1.0])
        logits2 = tensor([1.0, 1.0])
        labels2 = tensor([1.0, 1.0])
        g1 = float(F.logloss_grad(logits1, labels1).item())
        g2 = float(F.logloss_grad(logits2, labels2)[0])
        assert g1 == pytest.approx(2 * g2, rel=1e-5)

    def test_opposite_labels_opposite_sign(self):
        # Gradient for label +1 and -1 should have opposite signs at x=0
        logits = tensor([0.0])
        g_pos = float(F.logloss_grad(logits, tensor([1.0])).item())
        g_neg = float(F.logloss_grad(logits, tensor([-1.0])).item())
        assert g_pos == pytest.approx(-g_neg, rel=1e-5)

    def test_returns_tensor(self):
        assert isinstance(F.logloss_grad(tensor([1.0]), tensor([1.0])), Tensor)

    def test_numerical_gradient_matches(self):
        # Finite difference check against logloss
        x_val = 0.5
        y_val = 1.0
        eps = 1e-5
        loss_plus = float(F.logloss(tensor([x_val + eps]), tensor([y_val])).item())
        loss_minus = float(F.logloss(tensor([x_val - eps]), tensor([y_val])).item())
        numerical = (loss_plus - loss_minus) / (2 * eps)
        analytical = float(F.logloss_grad(tensor([x_val]), tensor([y_val])).item())
        assert analytical == pytest.approx(numerical, rel=1e-4)


# ── linear ────────────────────────────────────────────────────────────────────


class TestLinear:
    def test_basic_forward(self):
        x = tensor([[1.0, 2.0]])  # (1, 2)
        weight = tensor([[1.0], [1.0]])  # (2, 1)
        bias = tensor([[0.0]])  # (1, 1)
        out = F.linear(x, weight, bias)
        assert_close(out, [[3.0]])

    def test_bias_added(self):
        x = tensor([[1.0, 0.0]])
        weight = tensor([[1.0], [0.0]])
        bias = tensor([[5.0]])
        out = F.linear(x, weight, bias)
        assert_close(out, [[6.0]])

    def test_batch_input(self):
        x = tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        weight = tensor([[1.0, 0.0], [0.0, 1.0]])  # (2, 2) identity
        bias = tensor([[0.0, 0.0]])
        out = F.linear(x, weight, bias)
        assert_close(out, [[1.0, 2.0], [3.0, 4.0]])

    def test_output_shape(self):
        x = tensor(np.ones((4, 3)))  # (4, 3)
        weight = tensor(np.ones((3, 2)))  # (3, 2)
        bias = tensor(np.zeros((1, 2)))  # (1, 2)
        out = F.linear(x, weight, bias)
        assert out.shape == (4, 2)

    def test_non_tensor_input_raises(self):
        weight = tensor([[1.0], [1.0]])
        bias = tensor([[0.0]])
        with pytest.raises(ValueError, match="motorch.Tensor"):
            F.linear(np.array([[1.0, 2.0]]), weight, bias)

    def test_shape_mismatch_raises(self):
        x = tensor([[1.0, 2.0, 3.0]])  # (1, 3)
        weight = tensor([[1.0], [1.0]])  # (2, 1) — mismatch
        bias = tensor([[0.0]])
        with pytest.raises(AssertionError):
            F.linear(x, weight, bias)

    def test_zero_weight(self):
        x = tensor([[1.0, 2.0]])
        weight = tensor([[0.0], [0.0]])
        bias = tensor([[3.0]])
        out = F.linear(x, weight, bias)
        assert_close(out, [[3.0]])

    def test_returns_tensor(self):
        x = tensor([[1.0]])
        weight = tensor([[2.0]])
        bias = tensor([[0.0]])
        assert isinstance(F.linear(x, weight, bias), Tensor)

    def test_known_values(self):
        # Manual: [[1,2]] @ [[3],[4]] + [[5]] = [[1*3 + 2*4]] + [[5]] = [[16]]
        x = tensor([[1.0, 2.0]])
        weight = tensor([[3.0], [4.0]])
        bias = tensor([[5.0]])
        out = F.linear(x, weight, bias)
        assert_close(out, [[16.0]])
