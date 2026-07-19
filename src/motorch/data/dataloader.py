"""Minibatch data loading utilities."""

import math

import numpy as np


class DataLoader:
    """Iterate over a dataset in minibatches."""

    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        drop_last=False,
        rng=None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = rng

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            if self.rng is None:
                np.random.default_rng().shuffle(indices)
            else:
                self.rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            yield self.dataset[batch_indices]
