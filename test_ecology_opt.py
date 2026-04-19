import numpy as np
from scipy import ndimage, spatial
import time

def original(landscape_matrix):
    total_cells = landscape_matrix.size
    habitat_cells = np.sum(landscape_matrix == 1)
    habitat_proportion = habitat_cells / total_cells if total_cells > 0 else 0
    labeled_patches, n_patches = ndimage.label(landscape_matrix == 1)

    patch_sizes = []
    patch_perimeters = []

    for patch_id in range(1, n_patches + 1):
        patch_mask = labeled_patches == patch_id
        patch_size = np.sum(patch_mask)
        patch_sizes.append(patch_size)

        eroded = ndimage.binary_erosion(patch_mask)
        perimeter = np.sum(patch_mask) - np.sum(eroded)
        patch_perimeters.append(perimeter)

    distance_threshold = 3
    connectivity = 0
    if n_patches > 1:
        centroids = []
        for patch_id in range(1, n_patches + 1):
            y, x = np.where(labeled_patches == patch_id)
            centroids.append([np.mean(y), np.mean(x)])

        centroids = np.array(centroids)
        distances = spatial.distance_matrix(centroids, centroids)
        connected_pairs = np.sum(distances < distance_threshold) - n_patches
        possible_pairs = n_patches * (n_patches - 1)
        connectivity = connected_pairs / possible_pairs if possible_pairs > 0 else 0

    return patch_sizes, patch_perimeters, connectivity

def optimized(landscape_matrix):
    total_cells = landscape_matrix.size
    habitat_mask = landscape_matrix == 1
    habitat_cells = np.sum(habitat_mask)
    habitat_proportion = habitat_cells / total_cells if total_cells > 0 else 0
    labeled_patches, n_patches = ndimage.label(habitat_mask)

    if n_patches == 0:
        patch_sizes = []
        patch_perimeters = []
    else:
        # Vectorized patch sizes
        patch_sizes_arr = np.bincount(labeled_patches.ravel())[1:]
        patch_sizes = patch_sizes_arr.tolist()

        # Vectorized perimeters
        eroded_global = ndimage.binary_erosion(habitat_mask)
        perimeter_mask = habitat_mask & ~eroded_global
        patch_perimeters_arr = np.bincount(labeled_patches[perimeter_mask])

        # bincount might return fewer elements if the last patch has 0 perimeter (impossible but still)
        # we pad it to n_patches + 1
        if len(patch_perimeters_arr) <= n_patches:
            patch_perimeters_arr = np.pad(patch_perimeters_arr, (0, n_patches + 1 - len(patch_perimeters_arr)))
        patch_perimeters = patch_perimeters_arr[1:].tolist()

    distance_threshold = 3
    connectivity = 0
    if n_patches > 1:
        # Vectorized centroids
        indices = np.arange(1, n_patches + 1)
        centroids = ndimage.center_of_mass(habitat_mask, labeled_patches, index=indices)
        centroids = np.array(centroids)

        # cKDTree for distances
        tree = spatial.cKDTree(centroids)
        # query_pairs is <=, so we use threshold - 1e-9 to simulate <
        pairs = tree.query_pairs(distance_threshold - 1e-9)
        # pairs are unique (i < j), so we multiply by 2 for both directions
        connected_pairs = len(pairs) * 2
        possible_pairs = n_patches * (n_patches - 1)
        connectivity = connected_pairs / possible_pairs if possible_pairs > 0 else 0

    return patch_sizes, patch_perimeters, connectivity

# Test
np.random.seed(42)
mat = np.random.choice([0, 1], size=(200, 200), p=[0.9, 0.1])

t0 = time.time()
res1 = original(mat)
t1 = time.time()
res2 = optimized(mat)
t2 = time.time()

print("Original time:", t1 - t0)
print("Optimized time:", t2 - t1)

if res1[0] == res2[0]:
    print("Patch sizes match")
else:
    print("Patch sizes DO NOT match")

if res1[1] == res2[1]:
    print("Patch perimeters match")
else:
    print("Patch perimeters DO NOT match")

if np.isclose(res1[2], res2[2]):
    print("Connectivity matches")
else:
    print("Connectivity DOES NOT match", res1[2], res2[2])
