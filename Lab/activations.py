import numpy as np


def relu(Z):
    return np.maximum(0, Z)


def relu_prime(Z):
    dZ = np.array(Z, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid(Z):
    # Clip Z to prevent overflow.
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))


def sigmoid_prime(Z):
    s = sigmoid(Z)
    return s * (1 - s)
