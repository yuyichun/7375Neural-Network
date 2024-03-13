from neural_network import MultilayerNeuralNetwork
from dataset import generate_dataset

def main():
    X, Y = generate_dataset(num_samples=1000, circle_radius=0.5)
    nn = MultilayerNeuralNetwork(layer_sizes=[2, 10, 8, 8, 4, 1])
    nn.train(X, Y, learning_rate=0.01, num_iterations=1000)
    accuracy = nn.evaluate(X, Y)
    print(f"Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
