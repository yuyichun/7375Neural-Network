import numpy as np

class Perceptron:
    def __init__(self, input_size, num_classes):
        self.weights = np.random.randn(input_size, num_classes) * 0.01
        self.bias = np.zeros((1, num_classes))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X):
        Z = np.dot(X, self.weights) + self.bias
        A = self.sigmoid(Z)
        return A

    def compute_loss(self, A, Y):
        m = Y.shape[0]
        loss = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return loss

    def backpropagation(self, X, Y, A):
        m = X.shape[0]
        dZ = A - Y
        dW = 1/m * np.dot(X.T, dZ)
        dB = 1/m * np.sum(dZ, axis=0, keepdims=True)
        return dW, dB

    def update_parameters(self, dW, dB, learning_rate):
        self.weights = self.weights - learning_rate * dW
        self.bias = self.bias - learning_rate * dB

    def train(self, X_train, Y_train, iterations, learning_rate):
        for i in range(iterations):
            A = self.forward_propagation(X_train)
            loss = self.compute_loss(A, Y_train)
            dW, dB = self.backpropagation(X_train, Y_train, A)
            self.update_parameters(dW, dB, learning_rate)
            if i % 100 == 0:
                print(f"Iteration {i}: Loss {loss:.4f}")

    def predict(self, X):
        A = self.forward_propagation(X)
        return np.argmax(A, axis=1)
