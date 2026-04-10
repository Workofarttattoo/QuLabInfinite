import time
import numpy as np

def original_split(X, y, features):
    best_feature = None
    best_threshold = None
    best_mse = float('inf')

    for feature in features:
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            mask = X[:, feature] <= threshold
            if np.sum(mask) == 0 or np.sum(~mask) == 0:
                continue

            left_mse = np.var(y[mask]) * np.sum(mask)
            right_mse = np.var(y[~mask]) * np.sum(~mask)
            mse = (left_mse + right_mse) / len(y)

            if mse < best_mse:
                best_mse = mse
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold, best_mse

def optimized_split(X, y, features):
    best_feature = None
    best_threshold = None
    best_mse = float('inf')
    n_total = len(y)

    for feature in features:
        x_feat = X[:, feature]

        # Sort indices
        sort_idx = np.argsort(x_feat)
        x_sorted = x_feat[sort_idx]
        y_sorted = y[sort_idx]

        # Find valid split indices (where value changes)
        split_mask = x_sorted[:-1] != x_sorted[1:]
        split_indices = np.where(split_mask)[0]

        if len(split_indices) == 0:
            continue

        # Compute cumulative sums
        cum_sum = np.cumsum(y_sorted)
        cum_sq_sum = np.cumsum(y_sorted ** 2)

        total_sum = cum_sum[-1]
        total_sq_sum = cum_sq_sum[-1]

        # Array of left sizes
        n_left = split_indices + 1
        n_right = n_total - n_left

        # Left and right sums
        sum_left = cum_sum[split_indices]
        sum_right = total_sum - sum_left

        sq_sum_left = cum_sq_sum[split_indices]
        sq_sum_right = total_sq_sum - sq_sum_left

        # Variances * N
        left_mse = sq_sum_left - (sum_left ** 2) / n_left
        right_mse = sq_sum_right - (sum_right ** 2) / n_right

        # We need total MSE divided by n_total, but we can just minimize the sum
        mses = (left_mse + right_mse) / n_total

        min_idx = np.argmin(mses)
        if mses[min_idx] < best_mse:
            best_mse = mses[min_idx]
            best_feature = feature
            best_threshold = x_sorted[split_indices[min_idx]]

    return best_feature, best_threshold, best_mse

np.random.seed(42)
N = 1000
K = 5
X = np.random.randn(N, 10)
y = X.dot(np.random.randn(10)) + np.random.randn(N)
features = np.arange(K)

t0 = time.time()
res1 = original_split(X, y, features)
t1 = time.time()
print("Original:", res1, f"Time: {t1-t0:.4f}s")

t0 = time.time()
res2 = optimized_split(X, y, features)
t1 = time.time()
print("Optimized:", res2, f"Time: {t1-t0:.4f}s")

# Ensure results match
assert res1[0] == res2[0]
assert res1[1] == res2[1]
assert np.isclose(res1[2], res2[2])
print("Results match!")
