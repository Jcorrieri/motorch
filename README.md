# MoTorch

MoTorch is a lightweight, NumPy-based deep learning library inspired by the PyTorch API.  It provides a Tensor container with automatic differentiation, neural network modules, optimization utilities, and loss functions for building and training models from scratch.

## Basic Usage

Build a model, run a forward pass, and backpropagate gradients:

```python
import motorch as mo
import motorch.nn as nn
import motorch.optim as optim

x = mo.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
y = mo.tensor([[1.0], [-1.0]])

model = nn.Linear(in_features=2, out_features=1)
activation = nn.ReLU()
criterion = nn.LogisticLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

logits = activation(model(x))
loss = criterion(logits, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Training Loop

A standard training loop with train/validation splits and gradient descent:

```python
loss_fn = nn.LogisticLoss()
optimizer = optim.SGD(model.parameters(), lr=0.8)

for epoch in range(num_epochs):
    optimizer.zero_grad()

    logits = model(train_x)
    loss = loss_fn(logits, train_y)
    loss.backward()
    optimizer.step()

    with mo.no_grad():
        val_logits = model(val_x)
        val_loss = loss_fn(val_logits, val_y)
```

## Modules:

MoTorch follows the PyTorch module pattern. Models are defined by subclassing nn.Module and implementing forward:

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 3)
        self.act1   = nn.ReLU()
        self.output = nn.Linear(3, 1)

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.output(x)
        return x

model = MLP()
```

Currently available modules (more to come):

- Layers: nn.Linear
- Activations: nn.Sigmoid, nn.AltSigmoid, nn.Sgn, nn.ReLU
- Loss: nn.LogisticLoss
- Optimizers: optim.SGD
- Data utilities: motorch.data.TensorDataset and motorch.data.DataLoader

## Examples

The examples/ directory contains working notebooks demonstrating MoTorch for classification and model training, including:

- `examples/binary_classifier.ipynb` - binary classification on a 2D dataset with manual weight initialization,
Glorot initialization, a full train/validation loop, and decision boundary visualization.
- `examples/mnist_mlp_classifier.ipynb` - scaffold for a future MNIST MLP classifier.
  It downloads and parses MNIST with the Python standard library, creates
  `TensorDataset` and `DataLoader` objects, and defines an MLP skeleton. The
  criterion and training loop are intentionally pending until a multiclass loss
  is available.

## Development

This project is built with `uv` for environment management and `pytest` for testing.

Run the test suite with:

```bash
uv run pytest
```

Run linting & formatting with `Ruff`:

```bash
uv run ruff format --check .
uv run ruff check .
```

## References

The implementation is inspired by the following references:

- NumPy array dispatch and ndarray behavior: https://numpy.org/doc/stable/reference/generated/numpy.ndarray
- NumPy array function dispatch: https://numpy.org/doc/stable/user/basics.dispatch.html
- PyTorch module structure: https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn
- PyTorch module base class reference: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py#L407
- PyTorch parameter initialization reference: https://github.com/pytorch/pytorch/blob/main/torch/nn/init.py
- Andrej Karpathy's micrograd: https://www.youtube.com/watch?v=VMj-3S1tku0&t=1624s
