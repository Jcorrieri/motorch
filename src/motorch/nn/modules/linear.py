import numpy as np
import textwrap


class Linear():
    def __init__(self, num_features, num_neurons):
        self.num_features = num_features
        self.num_neurons = num_neurons

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        string = textwrap.dedent(
            f"""Linear(num_features={self.num_features}, num_neurons={self.num_neurons})
          Weights: {", ".join([str(weight) for weight in self.weights.T])}
          Biases: {", ".join([str(bias) for bias in self.biases.T])}
        """)
        return string

    def set_weights(self, weights: np.ndarray, biases: np.ndarray):
        self.weights = weights
        self.biases = biases

        expected_shape = (self.num_features, self.num_neurons)

        assert self.weights.shape == expected_shape,\
        f"Expected weights.shape = {expected_shape}, got {self.weights.shape}."

        expected_shape = (1, self.num_neurons)

        assert self.biases.shape == expected_shape,\
        f"Expected biases.shape = {expected_shape}, got {self.biases.shape}."

    def forward(self, x):
        assert self.weights is not None, "Weights must be uninitialized."

        expected_shape = (self.weights.shape)

        assert x.shape[-1] == expected_shape[0],\
        f"Expected x.shape = (m, {expected_shape[0]}), got {x.shape}."

        self.x = x

        return x @ self.weights + self.biases

