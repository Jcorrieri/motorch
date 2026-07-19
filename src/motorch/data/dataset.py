"""Dataset containers for MoTorch."""

import motorch as mo
from motorch import Tensor


class TensorDataset:
    """Dataset wrapping tensors with a shared first dimension."""

    def __init__(self, *tensors):
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor.")

        self.tensors = tuple(
            item if isinstance(item, Tensor) else mo.tensor(item) for item in tensors
        )
        expected_length = len(self.tensors[0])

        for item in self.tensors[1:]:
            if len(item) != expected_length:
                raise ValueError(
                    "All tensors must have the same size in the first dimension."
                )

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, index):
        return tuple(item[index] for item in self.tensors)
