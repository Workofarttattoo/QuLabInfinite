import numpy as np
from ecology_lab import EcologyLab
from scipy import spatial, ndimage
import time

def optimized_habitat_fragmentation(lab, landscape: np.ndarray, cell_size: float = 1.0):
    # Same initial logic to get labeled_patches and n_patches
    habitat_mask = landscape > 0
    labeled_patches, n_patches = ndimage.label(habitat_mask)

    # Calculate centroids efficiently using ndimage.center_of_mass
    if n_patches > 1:
        centroids = ndimage.center_of_mass(habitat_mask, labeled_patches, index=np.arange(1, n_patches + 1))
        centroids = np.array(centroids)

        distance_threshold = 3
        # Use cKDTree instead of distance_matrix
        tree = spatial.cKDTree(centroids)
        # cKDTree.query_pairs uses <= threshold, offset threshold to simulate <
        pairs = tree.query_pairs(distance_threshold - 1e-9)
        connected_pairs = len(pairs) * 2 # query_pairs returns unique pairs, times 2 for symmetric

        possible_pairs = n_patches * (n_patches - 1)
        connectivity = connected_pairs / possible_pairs if possible_pairs > 0 else 0
        return connectivity
    return 0

def main():
    np.random.seed(42)
    n = 200
    grid = np.random.choice([0, 1], size=(n, n), p=[0.7, 0.3])

    lab = EcologyLab()

    start_time = time.time()
    orig_metrics = lab.habitat_fragmentation_analysis(grid, cell_size=10.0)
    orig_time = time.time() - start_time

    start_time = time.time()
    # Replace logic in class to test properly
    # But let's just test our optimization logic here first
    habitat_mask = grid > 0
    labeled_patches, n_patches = ndimage.label(habitat_mask)
    if n_patches > 1:
        centroids = ndimage.center_of_mass(habitat_mask, labeled_patches, index=np.arange(1, n_patches + 1))
        centroids = np.array(centroids)
        distance_threshold = 3
        tree = spatial.cKDTree(centroids)
        pairs = tree.query_pairs(distance_threshold - 1e-9)
        connected_pairs = len(pairs) * 2
        possible_pairs = n_patches * (n_patches - 1)
        connectivity = connected_pairs / possible_pairs if possible_pairs > 0 else 0

    new_time = time.time() - start_time

    print(f"Original connectivity: {orig_metrics['connectivity']} (in {orig_time:.4f}s)")
    print(f"Optimized connectivity: {connectivity} (in {new_time:.4f}s)")

if __name__ == "__main__":
    main()
