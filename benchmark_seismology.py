import time
import numpy as np
from seismology_lab import SeismologyLab, SoilClass

lab = SeismologyLab()

# Benchmark site amplification (has a loop over frequencies)
freqs = np.linspace(0.01, 20.0, 100000)
input_motion = np.random.randn(100000)

start_time = time.time()
res = lab.site_amplification(input_motion, freqs, SoilClass.C)
end_time = time.time()
print(f"site_amplification time: {end_time - start_time:.4f} seconds")

# Benchmark dispersion
periods = np.linspace(0.1, 100.0, 100000)
layer_model = {'layer1_h': 10, 'layer1_vs': 2.0, 'layer2_vs': 3.0}
start_time = time.time()
res = lab.surface_wave_dispersion(periods, layer_model)
end_time = time.time()
print(f"dispersion time: {end_time - start_time:.4f} seconds")

# Benchmark GMPE
start_time = time.time()
res = lab.ground_motion_prediction(7.0, 50.0)
end_time = time.time()
print(f"GMPE time: {end_time - start_time:.4f} seconds")
