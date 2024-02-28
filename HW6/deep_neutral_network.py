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

    
    def generate_mini_batches(self, X, Y, mini_batch_size):
        m = X.shape[1]  # number of training examples
        mini_batches = []
        
        # Shuffle (X, Y)
        permutation = np.random.permutation(m)
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1,m))

        # Partition (shuffled_X, shuffled_Y) into mini-batches
        num_complete_minibatches = m // mini_batch_size

        for k in range(num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    
        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches

    def train(self, X, Y, iterations, learning_rate, mini_batch_size):
        for i in range(iterations):
            mini_batches = self.generate_mini_batches(X, Y, mini_batch_size)
            
            for mini_batch in mini_batches:
                (mini_batch_X, mini_batch_Y) = mini_batch
                
                # Forward propagation
                AL, caches = self.forward_propagation(mini_batch_X)

                # Compute cost
                cost = self.compute_cost(AL, mini_batch_Y)

                # Backward propagation
                grads = self.backward_propagation(AL, mini_batch_Y, caches)

                # Update parameters
                self.update_parameters(grads, learning_rate)
                
            # You can print the cost here every 100 iterations, for example
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
