import numpy as np
import time
from machine_learning_lab import MachineLearningLab, MLConfig

np.random.seed(42)
X = np.random.randn(1000, 100)
true_theta = np.random.randn(100)
# Make it sparse
true_theta[np.random.choice(100, 80, replace=False)] = 0
y = X.dot(true_theta) + np.random.randn(1000) * 0.1

lab = MachineLearningLab(MLConfig(epochs=100))

# OPTIMIZED VERSION
def lasso_optimized(X, y, alpha=0.1):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)

    # Precompute feature norms squared / n_samples
    z = np.sum(X ** 2, axis=0) / n_samples

    # Initial residual (since theta is zeros, residual is y)
    residual = y.copy()

    for _ in range(100):
        for j in range(n_features):
            if z[j] == 0:
                continue

            # Store old theta_j
            theta_j_old = theta[j]

            # Add feature j's contribution back to residual
            residual += X[:, j] * theta_j_old

            # Compute rho
            rho = X[:, j].dot(residual)

            # Soft thresholding
            if rho < -alpha/2:
                theta[j] = (rho + alpha/2) / z[j]
            elif rho > alpha/2:
                theta[j] = (rho - alpha/2) / z[j]
            else:
                theta[j] = 0

            # Remove feature j's new contribution from residual
            residual -= X[:, j] * theta[j]

    return theta

start = time.time()
theta_orig = lab.lasso_coordinate_descent(X, y, alpha=0.1)
time_orig = time.time() - start

start = time.time()
theta_opt = lasso_optimized(X, y, alpha=0.1)
time_opt = time.time() - start

print(f"Original Time: {time_orig:.4f}s")
print(f"Optimized Time: {time_opt:.4f}s")
print(f"Speedup: {time_orig / time_opt:.2f}x")
print(f"Difference: {np.linalg.norm(theta_orig - theta_opt):.6e}")
