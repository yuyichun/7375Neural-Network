import numpy as np;

class Parameter:
    def __init__(self):
        self.weights = None
        self.bias = None
        
    def set_bias(self,bias):
        self.bias = bias
    
    def get_weights(self):
        return self.weights
    
    def get_bias(self):
        return self.bias
    
class Neuron:
    def __init__(self,input_size):
        self.weights = None
        self.bias = None
        self.input_size = input_size
        self_agg_signal = None
        self.activation = None
        self.output = None
        
    
input_size = 2
num_neuron = 3
parameter = Parameter()
parameter.set_bias(np.random.randn(input_size))

inputs = np.random.randint(2, size=(input_size, num_neuron))
print("Inputs:", inputs)
        