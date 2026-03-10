import numpy as np

distances = np.random.rand(10, 10)
clusters = [[0, 2, 4], [1, 3, 5]]

i = 0
j = 1

avg_dist_orig = np.mean([distances[a, b] for a in clusters[i] for b in clusters[j]])
avg_dist_opt = np.mean(distances[np.ix_(clusters[i], clusters[j])])

print(f"Original: {avg_dist_orig}")
print(f"Optimized: {avg_dist_opt}")
assert np.isclose(avg_dist_orig, avg_dist_opt)
print("Assertion passed.")
