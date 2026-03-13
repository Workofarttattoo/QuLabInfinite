import time
import numpy as np
from machine_learning_lab import MachineLearningLab, MLConfig

def benchmark():
    lab = MachineLearningLab()

    np.random.seed(42)
    n_samples, n_features = 1000, 15
    X = np.random.randn(n_samples, n_features)

    start = time.time()
    lab.polynomial_features(X, degree=3)
    print(f"Time taken: {time.time() - start:.4f}s")

benchmark()
