import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def test_convolution():
    # Setup
    image = np.random.rand(512, 512)
    kernel = np.random.rand(3, 3)
    stride = 1

    h, w = image.shape
    kh, kw = kernel.shape
    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1

    # Original slow method
    start = time.time()
    output_slow = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            y = i * stride
            x = j * stride
            output_slow[i, j] = np.sum(image[y:y+kh, x:x+kw] * kernel)
    slow_time = time.time() - start

    # Optimized method
    start = time.time()
    windows = sliding_window_view(image, (kh, kw))
    if stride > 1:
        windows = windows[::stride, ::stride]
    output_fast = np.tensordot(windows, kernel, axes=([2, 3], [0, 1]))
    fast_time = time.time() - start

    print(f"Convolution: Slow = {slow_time:.4f}s, Fast = {fast_time:.4f}s, Speedup = {slow_time/fast_time:.2f}x")
    print(f"Match: {np.allclose(output_slow, output_fast)}")

def test_max_pool():
    # Setup
    image = np.random.rand(512, 512)
    pool_size = 2
    stride = 2

    h, w = image.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    # Original slow method
    start = time.time()
    output_slow = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            y = i * stride
            x = j * stride
            output_slow[i, j] = np.max(image[y:y+pool_size, x:x+pool_size])
    slow_time = time.time() - start

    # Optimized method
    start = time.time()
    windows = sliding_window_view(image, (pool_size, pool_size))
    if stride > 1:
        windows = windows[::stride, ::stride]
    output_fast = windows.max(axis=(2, 3))
    fast_time = time.time() - start

    print(f"Max Pool: Slow = {slow_time:.4f}s, Fast = {fast_time:.4f}s, Speedup = {slow_time/fast_time:.2f}x")
    print(f"Match: {np.allclose(output_slow, output_fast)}")

if __name__ == '__main__':
    # Using python3 -m py_compile as proxy for verifying env? No wait, this script just uses numpy.
    # Let's run it using a mock of numpy if numpy is missing.
    try:
        import numpy
        test_convolution()
        test_max_pool()
    except ImportError:
        print("Numpy not found!")
