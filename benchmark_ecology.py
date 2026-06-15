import numpy as np
from ecology_lab import EcologyLab
import time

def main():
    np.random.seed(42)
    # Generate a large random landscape
    n = 200
    grid = np.random.choice([0, 1], size=(n, n), p=[0.7, 0.3])

    lab = EcologyLab()

    start_time = time.time()
    metrics = lab.habitat_fragmentation_analysis(grid, cell_size=10.0)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.4f} seconds")
    print(f"Number of patches: {metrics['number_of_patches']}")

if __name__ == "__main__":
    main()
