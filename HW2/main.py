from generate_img import generate_img
from perceptron import Perceptron

def main():

    generate_img()
  
    input_size = 20 * 20  
    num_classes = 10 
    perceptron = Perceptron(input_size, num_classes)

    iterations = 1000
    learning_rate = 0.01
    perceptron.train(X_train, Y_train, iterations, learning_rate)
  
if __name__ == "__main__":
    main()
