# made by mike

import numpy as np


class Neuron:
    def __init__(self, X: np.ndarray, Y):
        self.weights = np.zeros((X.shape[1], 1))

    def train_one(self, x: np.ndarray, y) -> bool:
        pred = self.predict(x)
        if pred == y:
            return False

        x: np.ndarray = np.reshape(x, self.weights.shape)
        x = x / x.sum()
        if y == 1:
            self.weights += x
        else:
            self.weights -= x
        return True

    def predict(self, x: np.ndarray):
        pred = x @ self.weights
        if pred > 0:
            return 1
        return 0
