import numpy as np

def sigmoid(x):
    x_clipped = np.clip(x, -700, 700) # to avoid numerical instability
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

