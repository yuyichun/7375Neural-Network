import numpy as np
from model import MultilayerNeuralNetwork
from layers import FCLayer
from activations import relu, relu_prime, sigmoid, sigmoid_prime
from dataset import generate_dataset, visualize_dataset


def main():
    X, Y = generate_dataset(num_samples=1000, circle_radius=0.5)

    # Initialize the neural network
    nn = MultilayerNeuralNetwork()

    # Add layers to the network: input layer, hidden layers, and output layer
    nn.add(FCLayer(X.shape[1], 10, relu, relu_prime))  # Input layer
    nn.add(FCLayer(10, 8, relu, relu_prime))          # Hidden layer 1
    nn.add(FCLayer(8, 8, relu, relu_prime))           # Hidden layer 2
    nn.add(FCLayer(8, 4, relu, relu_prime))           # Hidden layer 3
    nn.add(FCLayer(4, 1, sigmoid, sigmoid_prime))     # Output layer

    # Train the network
    nn.fit(X, Y, epochs=1000, learning_rate=0.01)

    # Evaluate the network
    accuracy = nn.evaluate(X, Y)
    print(f"Accuracy: {accuracy:.2%}")

    # Visualize the dataset
    visualize_dataset(X, Y)


if __name__ == "__main__":
    main()


def main():
    X, Y = generate_dataset(num_samples=1000, circle_radius=0.5)
    input_size = X.shape[1]  # Number of input features
    output_size = 1  # Binary classification

    # Instantiate the neural network
    nn = MultilayerNeuralNetwork()

    # Add layers to the network
    nn.add(FCLayer(input_size, 10, relu, relu_prime))
    nn.add(FCLayer(10, 8, relu, relu_prime))
    nn.add(FCLayer(8, 8, relu, relu_prime))
    nn.add(FCLayer(8, 4, relu, relu_prime))
    nn.add(FCLayer(4, output_size, sigmoid, sigmoid_prime))

    # Train the network
    nn.fit(X, Y, epochs=1000, learning_rate=0.01)

    # Evaluate the network
    accuracy = nn.evaluate(X, Y)
    print(f"Accuracy: {accuracy:.2%}")

    # Visualize the dataset
    visualize_dataset(X, Y)


if __name__ == "__main__":
    main()
