"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

NEURAL NETWORKS LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi

@dataclass
class NeuralNetwork:
    input_size: int
    hidden_sizes: list[int]
    output_size: int
    learning_rate: float = 0.1

    def __post_init__(self):
        self.weights = []
        self.biases = []
        layers = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / (layers[i] + layers[i+1]))
            b = np.zeros((1, layers[i+1]), dtype=np.float64)
            self.weights.append(w.astype(np.float64))
            self.biases.append(b)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_pass(self, X):
        self.activations = [X]
        self.z_values = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.activations[-1], w) + b
            a = self.sigmoid(z)
            self.activations.append(a)
            self.z_values.append(z)

    def backward_pass(self, y_true, loss_derivative=None):
        delta = (self.activations[-1] - y_true) * self.sigmoid_derivative(self.z_values[-1])
        d_weights = [np.dot(self.activations[-2].T, delta)]
        d_biases = [delta]
        for w, z in reversed(list(zip(self.weights[:-1], self.z_values[:-1]))):
            delta = np.dot(delta, w.T) * self.sigmoid_derivative(z)
            d_weights.insert(0, np.dot(self.activations[-3 - i].T, delta))
            d_biases.insert(0, delta)
        for i in range(len(d_weights)):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * np.sum(d_biases[i], axis=0, keepdims=True)

    def train(self, X_train, y_train, epochs):
        for epoch in range(epochs):
            for i in range(len(X_train)):
                x = X_train[i].reshape((1, -1))
                y = y_train[i].reshape((1, -1))
                self.forward_pass(x)
                self.backward_pass(y)

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            x = np.array([x], dtype=np.float64)
            self.forward_pass(x)
            predictions.append(self.activations[-1])
        return np.vstack(predictions).round()

def run_demo():
    nn = NeuralNetwork(input_size=2, hidden_sizes=[3, 3], output_size=1, learning_rate=0.1)
    X_train = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    y_train = [[0], [1], [1], [0]]
    nn.train(X_train, y_train, epochs=5000)
    X_test = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    predictions = nn.predict(X_test)
    print(f"Predictions: {predictions}")

if __name__ == '__main__':
    run_demo()