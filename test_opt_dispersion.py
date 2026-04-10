import numpy as np
import time

def orig(periods, h1, vs1, vs2):
    rayleigh_velocities = []
    for T in periods:
        wavelength = T * vs1  # Approximate wavelength

        if wavelength < h1:
            # Short period - sensitive to shallow layer
            c_rayleigh = 0.92 * vs1
        elif wavelength > 4 * h1:
            # Long period - sensitive to deeper layer
            c_rayleigh = 0.92 * vs2
        else:
            # Transition zone
            weight = (wavelength - h1) / (3 * h1)
            c_rayleigh = 0.92 * (vs1 * (1 - weight) + vs2 * weight)

        rayleigh_velocities.append(c_rayleigh)

    return rayleigh_velocities

def opt(periods, h1, vs1, vs2):
    # Vectorized
    wavelength = periods * vs1

    mask_short = wavelength < h1
    mask_long = wavelength > 4 * h1
    mask_trans = ~(mask_short | mask_long)

    c_rayleigh = np.zeros_like(periods)
    c_rayleigh[mask_short] = 0.92 * vs1
    c_rayleigh[mask_long] = 0.92 * vs2

    weight = (wavelength[mask_trans] - h1) / (3 * h1)
    c_rayleigh[mask_trans] = 0.92 * (vs1 * (1 - weight) + vs2 * weight)

    return c_rayleigh.tolist()

periods = np.linspace(0.1, 100.0, 100000)
h1 = 10
vs1 = 2.0
vs2 = 3.0

t0 = time.time()
res1 = orig(periods, h1, vs1, vs2)
t1 = time.time()
time_orig = t1 - t0

t0 = time.time()
res2 = opt(periods, h1, vs1, vs2)
t1 = time.time()
time_opt = t1 - t0

print(f"Orig dispersion: {time_orig:.4f}s")
print(f"Opt dispersion:  {time_opt:.4f}s")
print(f"Speedup: {time_orig/time_opt:.2f}x")
print(f"Equal: {np.allclose(res1, res2)}")
