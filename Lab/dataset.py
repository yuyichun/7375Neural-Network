import numpy as np
import matplotlib.pyplot as plt


def generate_dataset(num_samples=1000, circle_radius=0.5):
    np.random.seed(1)
    X = np.random.rand(num_samples, 2) * 2 - 1
    Y = np.sqrt((X[:, 0] - 0)**2 + (X[:, 1] - 0)**2)
    # Make sure Y is a 2D column vector
    Y = (Y > circle_radius).astype(int).reshape(-1, 1)
    return X, Y


def visualize_dataset(X, Y):
    plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap='viridis', marker='.')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Synthetic Dataset for Binary Classification')
    plt.show()
