import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


class FCLayer(Layer):
    def __init__(self, input_size, output_size, activation_function, activation_function_prime):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))
        self.activation_function = activation_function
        self.activation_function_prime = activation_function_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation_function(
            np.dot(self.weights, input_data) + self.bias)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(self.weights.T, output_error)
        weights_error = np.dot(output_error, self.input.T)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * \
            np.sum(output_error, axis=1, keepdims=True)
        return input_error
