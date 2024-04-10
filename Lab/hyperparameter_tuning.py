import numpy as np
from model import MultilayerNeuralNetwork
from layers import FCLayer
from activations import relu, relu_prime, sigmoid, sigmoid_prime
from dataset import generate_dataset
from sklearn.model_selection import ParameterSampler

# Define a range for the hyperparameters
param_dist = {
    'learning_rate': np.logspace(-6, -3, 10),
    'first_layer_neurons': [10, 20, 30, 40, 50],
    'second_layer_neurons': [5, 10, 15, 20, 25],
}

# Generate the dataset
X, Y = generate_dataset(num_samples=1000, circle_radius=0.5)

# Specify the number of iterations for random search
n_iter = 10

# Track the best hyperparameters and their performance
best_params = None
best_accuracy = 0

# Perform the random search
for params in ParameterSampler(param_dist, n_iter):
    # Initialize the neural network
    nn = MultilayerNeuralNetwork()

    # Add layers to the network based on the sampled parameters
    nn.add(
        FCLayer(X.shape[1], params['first_layer_neurons'], relu, relu_prime))
    nn.add(FCLayer(params['first_layer_neurons'],
           params['second_layer_neurons'], relu, relu_prime))
    nn.add(FCLayer(params['second_layer_neurons'], 1, sigmoid, sigmoid_prime))

    # Train the network
    nn.fit(X, Y, epochs=1000, learning_rate=params['learning_rate'])

    # Evaluate the network
    accuracy = nn.evaluate(X, Y)

    # Check if the current parameters are better than the previous best
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

    # Print the results of the current iteration
    print(f"Params: {params}, Accuracy: {accuracy:.2%}")

# Print the best parameters and accuracy found during the search
print(f"Best params: {best_params}, Best accuracy: {best_accuracy:.2%}")
