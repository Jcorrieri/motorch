"""Linear module implementation for MoTorch."""

import motorch as mo
from motorch.tensor import tensor, tensor_zeros_like
import motorch.nn.functional as F
from motorch.nn.parameter import Parameter
from motorch.utils.no_grad import no_grad
from .module import Module


class Linear(Module):
    """A fully connected layer with weight and bias parameters."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(mo.empty(shape=(in_features, out_features)))
        self.bias = Parameter(mo.empty(shape=(1, out_features)))

    def forward(self, x):
        """Compute the layer output for input ``x``.

        This method validates the expected shapes for the weight and bias tensors
        before applying the linear transformation.
        """
        assert self.weight.shape == (self.in_features, self.out_features),\
            f"Unexpected weight shape {self.weight.shape}, expected\
            {(self.in_features, self.out_features)}."

        assert self.bias.shape == (1, self.out_features), \
            f"Unexpected bias shape {self.bias.shape}, expected {(1, self.out_features)}."

        with no_grad():
            out = F.linear(x, self.weight, self.bias)

        result = tensor(out.data)
        if no_grad._grad_enabled:
            result.grad_fn = lambda _x=x: self.grad_fn(result, _x)
            result._children = [x, self.weight, self.bias]
            result.requires_grad = True

        return result

    def grad_fn(self, result, x):
        with no_grad():
            for item in [x, self.weight, self.bias]:
                if item.grad is None:
                    item.grad = tensor_zeros_like(item)
            x.grad += result.grad @ self.weight.T
            self.weight.grad += x.T @ result.grad
            self.bias.grad += result.grad.mean(axis=0)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
