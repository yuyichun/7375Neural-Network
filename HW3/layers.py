import numpy as np

class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.output = None
        self.input = None
        self.delta = None

    def forward(self, input_data):
        self.input = input_data
        self.output = Activation.sigmoid(np.dot(self.weights, self.input) + self.bias)
        return self.output

    def backward(self, output_error, learning_rate):
        self.delta = output_error * Activation.sigmoid_derivative(self.output)
        weight_update = np.dot(self.delta, self.input.T)
        self.weights -= learning_rate * weight_update
        self.bias -= learning_rate * self.delta
