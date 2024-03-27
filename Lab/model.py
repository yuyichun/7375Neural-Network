import numpy as np
from layers import FCLayer
from activations import relu, relu_prime, sigmoid, sigmoid_prime
from losses import loss, loss_prime


class MultilayerNeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        result = input_data
        for layer in self.layers:
            result = layer.forward_propagation(result)
        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        for i in range(epochs):
            err = 0
            output = self.predict(x_train.T)
            err += loss(y_train.T, output)
            error = loss_prime(y_train.T, output)
            for layer in reversed(self.layers):
                error = layer.backward_propagation(error, learning_rate)
            err /= x_train.shape[0]
            if i % 100 == 0:
                print('epoch %d/%d   error=%f' % (i+1, epochs, err))

    def evaluate(self, x_test, y_test):
        predictions = self.predict(x_test.T)
        predictions = predictions > 0.5
        accuracy = np.mean(predictions == y_test.T)
        return accuracy
