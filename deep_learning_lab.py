"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

DEEP LEARNING LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List

# Constants and configuration
LEARNING_RATE = 0.1
EPOCHS = 100
ACTIVATION_FUNC = 'sigmoid'
LOSS_FUNCTION = 'mse'


@dataclass
class NeuralNetwork:
    layers: List[int]
    weights: List[np.ndarray] = field(init=False)
    biases: List[np.ndarray] = field(init=False)

    def __post_init__(self):
        self.weights = [np.random.randn(j, i) for j, i in zip(self.layers[1:], self.layers[:-1])]
        self.biases = [np.random.randn(j, 1) for j in self.layers[1:]]

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred)**2)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        activations = [x]
        zs = []
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = NeuralNetwork.sigmoid(z)
            activations.append(a)
        return a, (activations[:-1], zs)

    def backward(self, y_true: np.ndarray, activations: List[np.ndarray], zs: List[np.ndarray]) -> None:
        delta_list = [2 * (y_true - activations[-1]) * activations[-1] * (1 - activations[-1])]
        for i in range(len(self.layers) - 2, 0, -1):
            w = self.weights[i]
            delta = np.dot(w.T, delta_list[0])
            delta_list.insert(0, delta)
        del_w = [LEARNING_RATE * np.outer(delta_list[j], activations[j]) for j in range(len(self.layers) - 1)]
        del_b = [LEARNING_RATE * delta for delta in delta_list]
        self.weights = [w + dw for w, dw in zip(self.weights, del_w)]
        self.biases = [b + db for b, db in zip(self.biases, del_b)]

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        for _ in range(EPOCHS):
            a, (activations, zs) = self.forward(x_train)
            self.backward(y_train, activations, zs)


def run_demo():
    nn = NeuralNetwork(layers=[2, 3, 1])
    x_train = np.array([[0.5], [0.8]])
    y_train = np.array([[0.9]])
    nn.train(x_train.T, y_train.T)
    output, _ = nn.forward(np.array([[[0.7]], [[0.6]]]))
    print(f"Output after training: {output[0][0]:.4f}")


if __name__ == '__main__':
    run_demo()