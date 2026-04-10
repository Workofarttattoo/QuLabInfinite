import time
import numpy as np
from machine_learning_lab import MachineLearningLab, MLConfig

def benchmark():
    lab = MachineLearningLab()

    np.random.seed(42)
    n_samples, n_features = 1000, 10
    X = np.random.randn(n_samples, n_features)
    y = X.dot(np.random.randn(n_features)) + np.random.randn(n_samples)

    start = time.time()
    lab.random_forest_regressor(X, y, n_trees=20, max_depth=6)
    print(f"Time taken: {time.time() - start:.4f}s")

benchmark()
