# Assuming both Layer and Activation classes are in layers.py

from layers import Layer, Activation
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Layer(input_size, hidden_size)
        self.output_layer = Layer(hidden_size, output_size)

    def forward(self, input_data):
        hidden_output = self.hidden_layer.forward(input_data)
        return self.output_layer.forward(hidden_output)

    def backward(self, input_data, output_error, learning_rate):
        output_delta = output_error * Activation.sigmoid_derivative(self.output_layer.output)
        hidden_error = np.dot(self.output_layer.weights.T, output_delta)
        
        self.output_layer.backward(output_delta, learning_rate)
        self.hidden_layer.backward(hidden_error, learning_rate)

    def train(self, input_data, target, learning_rate):
        output = self.forward(input_data)
        output_error = target - output
        self.backward(input_data, output_error, learning_rate)

    def predict(self, input_data):
        return self.forward(input_data)
