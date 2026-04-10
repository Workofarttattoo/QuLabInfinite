"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MACHINE LEARNING LAB
Free gift to the scientific community from QuLabInfinite.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import List

@dataclass
class MLConfig:
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    input_dim: int = field(default=1)
    hidden_dim: int = field(default=64)
    output_dim: int = field(default=1)

class MachineLearningModel:
    def __init__(self, config: MLConfig):
        self.config = config
        self.weights_input_hidden = np.random.randn(config.input_dim, config.hidden_dim).astype(np.float64)
        self.bias_input_hidden = np.zeros((1, config.hidden_dim)).astype(np.float64)
        self.weights_hidden_output = np.random.randn(config.hidden_dim, config.output_dim).astype(np.float64)
        self.bias_hidden_output = np.zeros((1, config.output_dim)).astype(np.float64)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def predict(self, X: np.ndarray) -> np.ndarray:
        hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
        hidden_layer_output = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_hidden_output
        return output_layer_input

    def train(self, X: List[np.ndarray], y: np.ndarray):
        for epoch in range(self.config.epochs):
            hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_input_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output) + self.bias_hidden_output

            output_error = y - output_layer_input
            output_delta = output_error * (output_layer_input * (1 - output_layer_input))

            hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
            hidden_delta = hidden_error * (hidden_layer_output * (1 - hidden_layer_output))

            self.weights_hidden_output += self.config.learning_rate * np.dot(hidden_layer_output.T, output_delta)
            self.bias_hidden_output += self.config.learning_rate * np.sum(output_delta, axis=0, keepdims=True)

            self.weights_input_hidden += self.config.learning_rate * np.dot(X.T, hidden_delta)
            self.bias_input_hidden += self.config.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

def run_demo():
    config = MLConfig(input_dim=2, output_dim=1)
    model = MachineLearningModel(config)
    
    X = [np.random.rand(1, 2).astype(np.float64) for _ in range(10)]
    y = np.array([model.predict(x)[0] + (np.random.rand() - 0.5) * 0.1 for x in X]).astype(np.float64)
    
    model.train(X=X, y=y)

if __name__ == '__main__':
    run_demo()