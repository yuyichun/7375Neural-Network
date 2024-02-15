import numpy as np
from activation_function import ActivationFunction


class DeepNeuralNetwork:
    def __init__(self, layers_dims):
        self.parameters = {}
        self.L = len(layers_dims) - 1  # number of layers in the network
        self.cache = {}

        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = np.random.randn(
                layers_dims[l], layers_dims[l - 1]) * 0.01
            self.parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    def forward_propagation(self, X):
        A = X
        caches = []
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            if l == self.L - 1:
                A = ActivationFunction.sigmoid(Z)
            else:
                A = ActivationFunction.relu(Z)
            caches.append((A_prev, W, b, Z))
        return A, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
        return np.squeeze(cost)

    def backpropagation(self, AL, Y, caches):
        grads = {}
        L = self.L
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        for l in reversed(range(1, L + 1)):
            current_cache = caches[l-1]
            A_prev, W, b, Z = current_cache
            if l == L:
                dZ = ActivationFunction.sigmoid_derivative(Z) * dAL
            else:
                dZ = ActivationFunction.relu_derivative(Z) * dAL

            dW = (1 / m) * np.dot(dZ, A_prev.T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dAL = np.dot(W.T, dZ)

            grads["dW" + str(l)] = dW
            grads["db" + str(l)] = db

        return grads

    def update_parameters(self, grads, learning_rate):
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] -= learning_rate * \
                grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * \
                grads['db' + str(l)]
