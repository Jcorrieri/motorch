from .linear import Linear


# TODO: Refactor to actual module class
class Model():
    def __init__(self):
        self.learnable = []

    def __call__(self, x):
        return self.forward(x)

    def __str__(self):
        string = "Model:\n-----------\n"

        for attr, value in self.__dict__.items():
            if attr == "learnable": continue
            string += f"{attr}: {str(value)}"

        return string

    def _init_learnable(self):
        for attr, _ in self.__dict__.items():
            layer = getattr(self, attr)

            if isinstance(layer, Linear): # only Linear layers have weights
                initialized = layer in self.learnable
                if not initialized:
                    self.learnable.append(layer)

    def init_weights(self, weights: list, biases: list):
        if not self.learnable:
            self._init_learnable()

        assert len(weights) == len(self.learnable),\
        f"Expected {len(self.learnable)} weight arrays, got {len(weights)}."

        assert len(biases) == len(self.learnable),\
        f"Expected {len(self.learnable)} bias arrays, got {len(biases)}."

        for i, layer in enumerate(self.learnable):
            layer.set_weights(weights[i], biases[i])

    def forward(self, x):
        pass
