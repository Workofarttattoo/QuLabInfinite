import numpy as np
import time

def orig(frequency, fv, fa, fundamental_freq, Q):
    amplification = np.ones_like(frequency)
    for i, f in enumerate(frequency):
        if f < 0.1:
            amplification[i] = fv
        elif f > 10:
            amplification[i] = fa
        else:
            # Resonance at fundamental frequency
            amplification[i] = 1 + (fa - 1) * fundamental_freq ** 2 / \
                              (fundamental_freq ** 2 + (f - fundamental_freq) ** 2 / Q ** 2)
    return amplification

def opt(frequency, fv, fa, fundamental_freq, Q):
    # Vectorized operations
    amplification = np.ones_like(frequency)

    mask_low = frequency < 0.1
    mask_high = frequency > 10
    mask_mid = ~(mask_low | mask_high)

    amplification[mask_low] = fv
    amplification[mask_high] = fa

    f_mid = frequency[mask_mid]
    amplification[mask_mid] = 1 + (fa - 1) * fundamental_freq ** 2 / \
                             (fundamental_freq ** 2 + (f_mid - fundamental_freq) ** 2 / Q ** 2)

    return amplification

frequency = np.linspace(0.01, 20.0, 100000)
fv = 1.7
fa = 1.2
fundamental_freq = 2.5
Q = 20

# Time orig
t0 = time.time()
res1 = orig(frequency, fv, fa, fundamental_freq, Q)
t1 = time.time()
time_orig = t1 - t0

# Time opt
t0 = time.time()
res2 = opt(frequency, fv, fa, fundamental_freq, Q)
t1 = time.time()
time_opt = t1 - t0

print(f"Orig: {time_orig:.4f}s")
print(f"Opt:  {time_opt:.4f}s")
print(f"Speedup: {time_orig/time_opt:.2f}x")
print(f"Equal: {np.allclose(res1, res2)}")
