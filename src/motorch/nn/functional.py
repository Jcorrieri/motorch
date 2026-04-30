"""
Functional components for MoTorch. PyTorch delegates these to underlying C logic, but for simplicity the 
explicit implementations are defined here, in Python.
"""
import motorch as mo

# --- Activations --- # 

def sigmoid(x):
    x_clipped = mo.clip(x, -700, 700) # to avoid numerical instability
    return 1 / (1 + mo.exp(-x_clipped))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# --- Layers --- #

def linear(x, weight, bias):
    if not isinstance(x, mo.Tensor):
        raise ValueError("Input must be of type motorch.Tensor")

    assert x.shape[-1] == weight.shape[0],\
        f"Expected x.shape = ({len(x)}, {weight.shape[0]}), got {x.shape}."

    return x @ weight + bias
