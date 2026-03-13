import time
import numpy as np
from machine_learning_lab import MachineLearningLab, MLConfig

def benchmark():
    config = MLConfig(epochs=50, random_state=42)
    lab = MachineLearningLab(config)

    np.random.seed(42)
    n_samples, n_features = 5000, 50
    X = np.random.randn(n_samples, n_features)
    y = X.dot(np.random.randn(n_features)) + np.random.randn(n_samples)

    start = time.time()
    lab.cross_validation(X, y, k_folds=10)
    print(f"Time taken: {time.time() - start:.4f}s")

benchmark()
