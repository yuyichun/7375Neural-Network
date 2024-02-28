import numpy as np

def normalize_inputs(inputs):

    mean = np.mean(inputs, axis=0)
    std = np.std(inputs, axis=0)
    std_replaced = np.where(std == 0, 1, std)
    normalized_inputs = (inputs - mean) / std_replaced
    return normalized_inputs
