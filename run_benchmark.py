import time
import numpy as np
from scipy import ndimage, spatial
import ecology_lab

lab = ecology_lab.EcologyLab()
np.random.seed(42)
# Create a large landscape to show performance difference
mat = np.random.choice([0, 1], size=(200, 200), p=[0.9, 0.1])

t0 = time.time()
res1 = lab.habitat_fragmentation_analysis(mat)
t1 = time.time()

print("Original Time:", t1 - t0)
print("Result size:", res1['number_of_patches'])
