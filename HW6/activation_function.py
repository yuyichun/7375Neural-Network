import numpy as np

class ActivationFunctions:
    @staticmethod
    def linear(z):
        return z

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return z > 0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def softmax(z):
        exps = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)
