"""Tests for DataLoader."""

import numpy as np
import pytest

import motorch as mo
from motorch.data import DataLoader, TensorDataset


def make_dataset(num_samples=5):
    features = mo.tensor(np.arange(num_samples * 2).reshape(num_samples, 2))
    labels = mo.tensor(np.arange(num_samples))
    return TensorDataset(features, labels)


class TestDataLoader:
    def test_rejects_non_positive_batch_size(self):
        with pytest.raises(ValueError, match="positive integer"):
            DataLoader(make_dataset(), batch_size=0)

    def test_len_counts_incomplete_batch_by_default(self):
        loader = DataLoader(make_dataset(num_samples=5), batch_size=2)

        assert len(loader) == 3

    def test_len_drops_incomplete_batch_when_drop_last(self):
        loader = DataLoader(make_dataset(num_samples=5), batch_size=2, drop_last=True)

        assert len(loader) == 2

    def test_batches_have_expected_shapes(self):
        loader = DataLoader(make_dataset(num_samples=5), batch_size=2)
        batches = list(loader)

        batch_features, batch_labels = batches[0]
        final_features, final_labels = batches[-1]

        assert batch_features.shape == (2, 2)
        assert batch_labels.shape == (2,)
        assert final_features.shape == (1, 2)
        assert final_labels.shape == (1,)

    def test_drop_last_omits_incomplete_final_batch(self):
        loader = DataLoader(make_dataset(num_samples=5), batch_size=2, drop_last=True)
        batches = list(loader)

        assert len(batches) == 2
        for features, labels in batches:
            assert features.shape == (2, 2)
            assert labels.shape == (2,)

    def test_without_shuffle_preserves_order(self):
        loader = DataLoader(make_dataset(num_samples=5), batch_size=3)
        first_batch_features, first_batch_labels = next(iter(loader))

        np.testing.assert_array_equal(
            first_batch_features.data, [[0, 1], [2, 3], [4, 5]]
        )
        np.testing.assert_array_equal(first_batch_labels.data, [0, 1, 2])

    def test_shuffle_is_reproducible_with_seeded_rng(self):
        first_loader = DataLoader(
            make_dataset(num_samples=8),
            batch_size=8,
            shuffle=True,
            rng=np.random.default_rng(42),
        )
        second_loader = DataLoader(
            make_dataset(num_samples=8),
            batch_size=8,
            shuffle=True,
            rng=np.random.default_rng(42),
        )

        _, first_labels = next(iter(first_loader))
        _, second_labels = next(iter(second_loader))

        np.testing.assert_array_equal(first_labels.data, second_labels.data)
        assert not np.array_equal(first_labels.data, np.arange(8))
