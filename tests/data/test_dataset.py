"""Tests for TensorDataset."""

import numpy as np
import pytest

import motorch as mo
from motorch import Tensor
from motorch.data import TensorDataset


class TestTensorDataset:
    def test_requires_at_least_one_tensor(self):
        with pytest.raises(ValueError, match="at least one tensor"):
            TensorDataset()

    def test_len_returns_first_dimension(self):
        dataset = TensorDataset(mo.tensor([[1, 2], [3, 4]]), mo.tensor([0, 1]))

        assert len(dataset) == 2

    def test_rejects_mismatched_first_dimensions(self):
        features = mo.tensor([[1, 2], [3, 4]])
        labels = mo.tensor([0, 1, 2])

        with pytest.raises(ValueError, match="same size"):
            TensorDataset(features, labels)

    def test_converts_non_tensor_inputs(self):
        dataset = TensorDataset([[1, 2], [3, 4]], [0, 1])
        features, labels = dataset[0]

        assert isinstance(features, Tensor)
        assert isinstance(labels, Tensor)
        np.testing.assert_array_equal(features.data, [1, 2])
        np.testing.assert_array_equal(labels.data, 0)

    def test_integer_indexing_returns_sample_tuple(self):
        dataset = TensorDataset(mo.tensor([[1, 2], [3, 4]]), mo.tensor([0, 1]))
        features, label = dataset[1]

        np.testing.assert_array_equal(features.data, [3, 4])
        np.testing.assert_array_equal(label.data, 1)

    def test_slice_indexing_returns_batched_tuple(self):
        dataset = TensorDataset(
            mo.tensor([[1, 2], [3, 4], [5, 6]]), mo.tensor([0, 1, 2])
        )
        features, labels = dataset[:2]

        np.testing.assert_array_equal(features.data, [[1, 2], [3, 4]])
        np.testing.assert_array_equal(labels.data, [0, 1])

    def test_index_array_returns_batched_tuple(self):
        dataset = TensorDataset(
            mo.tensor([[1, 2], [3, 4], [5, 6]]), mo.tensor([0, 1, 2])
        )
        features, labels = dataset[np.array([2, 0])]

        np.testing.assert_array_equal(features.data, [[5, 6], [1, 2]])
        np.testing.assert_array_equal(labels.data, [2, 0])
