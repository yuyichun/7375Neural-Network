import numpy as np

epsilon = 1e-15


def loss(y_true, y_pred):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    m = y_true.shape[1]
    cost = -np.sum(y_true * np.log(y_pred) +
                   (1 - y_true) * np.log(1 - y_pred)) / m
    return cost


def loss_prime(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred))
