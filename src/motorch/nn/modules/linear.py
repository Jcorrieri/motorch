import textwrap

import motorch as mo
from .module import Module
import motorch.nn.functional as F


class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = mo.empty(shape=(in_features, out_features))
        self.bias = mo.empty(shape=(1, out_features))

    def __str__(self):
        string = textwrap.dedent(
            f"""Linear(in_features={self.in_features}, out_features={self.out_features})
          Weights: {", ".join([str(weight) for weight in self.weight.T])}
          Biases: {", ".join([str(bias) for bias in self.bias.T])}
        """)
        return string

    def forward(self, x):
        self.x = x

        assert self.weight.shape == (self.in_features, self.out_features),\
            f"Unexpected weight shape {self.weight.shape}, expected\
            {(self.in_features, self.out_features)}."

        assert self.bias.shape == (1, self.out_features), \
            f"Unexpected bias shape {self.bias.shape}, expected {(1, self.out_features)}."

        return F.linear(x, self.weight, self.bias)

