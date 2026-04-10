import time
import numpy as np
from machine_learning_lab import MachineLearningLab, MLConfig

def benchmark():
    config = MLConfig(epochs=50, batch_size=32, random_state=42)
    lab = MachineLearningLab(config)

    # Large dataset
    np.random.seed(42)
    n_samples, n_features = 100000, 50
    X = np.random.randn(n_samples, n_features)
    y = X.dot(np.random.randn(n_features)) + np.random.randn(n_samples)

    start = time.time()
    lab.stochastic_gradient_descent(X, y)
    print(f"Time taken: {time.time() - start:.4f}s")

benchmark()
