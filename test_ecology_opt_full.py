import numpy as np
from ecology_lab import EcologyLab
from scipy import spatial, ndimage
import time

def optimized_habitat_fragmentation(self, landscape: np.ndarray, cell_size: float = 1.0):
    total_cells = landscape.size
    habitat_mask = landscape > 0
    habitat_cells = np.sum(habitat_mask)

    habitat_proportion = habitat_cells / total_cells if total_cells > 0 else 0

    # Identify individual patches
    labeled_patches, n_patches = ndimage.label(habitat_mask)

    if n_patches == 0:
        return {
            'habitat_proportion': habitat_proportion,
            'number_of_patches': 0,
            'mean_patch_size': 0,
            'largest_patch_index': 0,
            'edge_density': 0,
            'total_edge_length': 0,
            'mean_shape_index': 1,
            'connectivity': 0,
            'fragmentation_index': 1.0,
            'patch_density': 0
        }

    # Patch metrics vectorized
    patch_sizes = np.bincount(labeled_patches.ravel())[1:]

    # Calculate perimeter (edge cells)
    eroded = ndimage.binary_erosion(habitat_mask)
    edge_mask = habitat_mask & ~eroded
    labeled_edges = labeled_patches * edge_mask
    patch_perimeters = np.bincount(labeled_edges.ravel())[1:]

    patch_sizes_area = patch_sizes * cell_size ** 2
    patch_perimeters_length = patch_perimeters * cell_size

    # Fragmentation metrics
    mean_patch_size = np.mean(patch_sizes_area) if len(patch_sizes_area) > 0 else 0
    largest_patch_index = np.max(patch_sizes_area) / (habitat_cells * cell_size ** 2) if habitat_cells > 0 else 0

    # Edge density
    total_edge = np.sum(patch_perimeters_length)
    edge_density = total_edge / (total_cells * cell_size ** 2) if total_cells > 0 else 0

    # Mean shape index (circle = 1, complex shape > 1)
    # Filter out size 0 which shouldn't happen but just in case
    expected_perimeters = 2 * np.sqrt(np.pi * patch_sizes_area)
    shape_indices = np.where(expected_perimeters > 0, patch_perimeters_length / expected_perimeters, 1)
    mean_shape_index = np.mean(shape_indices) if len(shape_indices) > 0 else 1

    # Connectivity (proportion of habitat within distance threshold)
    distance_threshold = 3  # cells
    connectivity = 0
    if n_patches > 1:
        centroids = ndimage.center_of_mass(habitat_mask, labeled_patches, index=np.arange(1, n_patches + 1))
        centroids = np.array(centroids)

        tree = spatial.cKDTree(centroids)
        pairs = tree.query_pairs(distance_threshold - 1e-9)
        connected_pairs = len(pairs) * 2
        possible_pairs = n_patches * (n_patches - 1)
        connectivity = connected_pairs / possible_pairs if possible_pairs > 0 else 0

    return {
        'habitat_proportion': habitat_proportion,
        'number_of_patches': n_patches,
        'mean_patch_size': mean_patch_size,
        'largest_patch_index': largest_patch_index,
        'edge_density': edge_density,
        'total_edge_length': total_edge,
        'mean_shape_index': mean_shape_index,
        'connectivity': connectivity,
        'fragmentation_index': 1 - largest_patch_index,
        'patch_density': n_patches / (total_cells * cell_size ** 2) if total_cells > 0 else 0
    }

def main():
    np.random.seed(42)
    n = 200
    grid = np.random.choice([0, 1], size=(n, n), p=[0.7, 0.3])

    lab = EcologyLab()

    start_time = time.time()
    orig_metrics = lab.habitat_fragmentation_analysis(grid, cell_size=10.0)
    orig_time = time.time() - start_time

    start_time = time.time()
    new_metrics = optimized_habitat_fragmentation(lab, grid, cell_size=10.0)
    new_time = time.time() - start_time

    print(f"Original time: {orig_time:.4f}s")
    print(f"Optimized time: {new_time:.4f}s")
    print(f"Speedup: {orig_time/new_time:.2f}x")

    for key in orig_metrics:
        print(f"{key}: {orig_metrics[key]} == {new_metrics[key]} -> {np.isclose(orig_metrics[key], new_metrics[key])}")

if __name__ == "__main__":
    main()
