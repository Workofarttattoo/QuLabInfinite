import time
import numpy as np
from machine_learning_lab import MachineLearningLab, MLConfig

def benchmark():
    config = MLConfig(epochs=500, random_state=42)
    lab = MachineLearningLab(config)

    np.random.seed(42)
    n_samples, n_features = 10000, 50
    X = np.random.randn(n_samples, n_features)
    y = X.dot(np.random.randn(n_features)) + np.random.randn(n_samples)

    start = time.time()
    lab.gradient_descent(X, y)
    print(f"Time taken: {time.time() - start:.4f}s")

benchmark()
