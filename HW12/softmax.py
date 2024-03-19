import numpy as np

def softmax_regression(X, theta):
    # Calculate the matrix product of X and theta
    logits = np.dot(X, theta)

    # Apply the softmax to logits
    # Subtract np.max for numerical stability (avoids potential overflow issues)
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probabilities = exps / np.sum(exps, axis=1, keepdims=True)

    return probabilities

M = 5  # number of examples
N = 3  # number of features
K = 4  # number of classes

X_example = np.random.rand(M, N)  # Feature matrix for M examples
theta_example = np.random.rand(N, K)  # Parameter matrix for K classes

# Call softmax regression function with example data
softmax_predictions = softmax_regression(X_example, theta_example)
print("Softmax Predictions:\n", softmax_predictions)
