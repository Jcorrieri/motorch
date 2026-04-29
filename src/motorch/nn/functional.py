import motorch as mo

def sigmoid(x):
    x_clipped = mo.clip(x, -700, 700) # to avoid numerical instability
    return 1 / (1 + mo.exp(-x_clipped))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

