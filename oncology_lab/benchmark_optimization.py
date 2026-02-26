
import time
import numpy as np

def original_laplacian(field, resolution):
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) +
        np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2) -
        6 * field
    ) / (resolution ** 2)

def optimized_laplacian(field, resolution):
    # Pad with wrap to handle periodic boundaries (implied by np.roll)
    padded = np.pad(field, 1, mode='wrap')

    # Slices for neighbors
    # axis 0
    start_0 = padded[:-2, 1:-1, 1:-1]  # roll +1
    end_0   = padded[2:,  1:-1, 1:-1]  # roll -1

    # axis 1
    start_1 = padded[1:-1, :-2, 1:-1]  # roll +1
    end_1   = padded[1:-1, 2:,  1:-1]  # roll -1

    # axis 2
    start_2 = padded[1:-1, 1:-1, :-2]  # roll +1
    end_2   = padded[1:-1, 1:-1, 2:]   # roll -1

    laplacian = (
        start_0 + end_0 +
        start_1 + end_1 +
        start_2 + end_2 -
        6 * field
    ) / (resolution ** 2)

    return laplacian

def benchmark():
    print("Benchmarking Laplacian implementations...")
    shape = (200, 200, 200)
    resolution = 10.0
    field = np.random.rand(*shape).astype(np.float32)

    # Warmup
    _ = original_laplacian(field, resolution)
    _ = optimized_laplacian(field, resolution)

    # Verify correctness
    res_orig = original_laplacian(field, resolution)
    res_opt = optimized_laplacian(field, resolution)

    diff = np.max(np.abs(res_orig - res_opt))
    if diff < 1e-5:
        print(f"✅ Correctness verified (max diff: {diff})")
    else:
        print(f"❌ Correctness FAILED (max diff: {diff})")
        return

    # Measure Original
    start_time = time.time()
    iterations = 20
    for _ in range(iterations):
        _ = original_laplacian(field, resolution)
    duration_orig = time.time() - start_time
    avg_orig = duration_orig / iterations
    print(f"Original: {avg_orig:.4f} s/iter")

    # Measure Optimized
    start_time = time.time()
    for _ in range(iterations):
        _ = optimized_laplacian(field, resolution)
    duration_opt = time.time() - start_time
    avg_opt = duration_opt / iterations
    print(f"Optimized: {avg_opt:.4f} s/iter")

    speedup = avg_orig / avg_opt
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark()
