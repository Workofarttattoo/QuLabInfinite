import time
import numpy as np
from astrophysics_lab import AstrophysicsLab

lab = AstrophysicsLab()
masses = np.array([1e30] * 100)
positions = np.random.rand(100, 3) * 1e11
velocities = np.random.rand(100, 3) * 3e4

start = time.time()
lab.nbody_gravitational_dynamics(masses, positions, velocities, 3600, 100)
end = time.time()
print('Time taken:', end - start)
