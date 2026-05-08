# MoTorch

MoTorch is a lightweight, NumPy-based deep learning library inspired by the PyTorch API. It provides a minimal Tensor container, simple neural network modules, optimization utilities, and loss functions for building small models and experimenting with training loops.

## Basic Usage

Create tensors, build a model, and run a forward pass with familiar primitives:

```python
import motorch as mo
from motorch.nn import Linear, Sigmoid, LogisticLoss
from motorch.optim import SGD

x = mo.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
model = Linear(in_features=2, out_features=1)
activation = Sigmoid()
criterion = LogisticLoss()
optimizer = SGD(model.parameters(), lr=0.01)

logits = activation(model(x))
loss = criterion(logits, mo.tensor([1, -1]))

optimizer.zero_grad()
# backward pass not implemented in this prototype
# optimizer.step()
```

## Examples

The `examples/` directory contains working notebooks that demonstrate how to use MoTorch for basic classification and model training. These examples show how to:

- construct simple feedforward models
- apply activation functions and loss modules
- run training loops with parameter updates

Start with `examples/binary_classifier.ipynb` to see a concrete binary classification workflow.

## Development

This project is built with `uv` for environment management and `pytest` for testing.

Run the test suite with:

```bash
uv run pytest
```

## References

The implementation is inspired by the following references:

- NumPy array dispatch and ndarray behavior: https://numpy.org/doc/stable/reference/generated/numpy.ndarray
- NumPy array function dispatch: https://numpy.org/doc/stable/user/basics.dispatch.html
- PyTorch module structure: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn
- PyTorch module base class reference: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L407
- PyTorch parameter initialization reference: https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py
