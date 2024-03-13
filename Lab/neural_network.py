import numpy as np

class MultilayerNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        np.random.seed(3)
        parameters = {}
        L = len(self.layer_sizes)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layer_sizes[l], 1))
        return parameters

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward_propagation(self, X):
        A = X
        caches = []
        L = len(self.parameters) // 2
        for l in range(1, L):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(W, A_prev) + b
            A = self.relu(Z)
            caches.append((A_prev, W, b, Z))
        W = self.parameters['W' + str(L)]
        b = self.parameters['b' + str(L)]
        Z = np.dot(W, A) + b
        AL = self.sigmoid(Z)
        caches.append((A, W, b, Z))
        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL)) / m
        return cost

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def sigmoid_backward(self, dA, Z):
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1-s)
        return dZ

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, "sigmoid")
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        return grads
    
    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for l in range(L):
            self.parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
            self.parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]

    def train(self, X, Y, learning_rate, num_iterations):
        for i in range(num_iterations):
            AL, caches = self.forward_propagation(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_propagation(AL, Y, caches)
            self.update_parameters(grads, learning_rate)
            if i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        predictions = AL > 0.5
        return predictions

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y)
        return accuracy