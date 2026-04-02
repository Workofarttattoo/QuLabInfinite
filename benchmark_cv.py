import time
import numpy as np
from computer_vision_lab import ComputerVisionLab

def run_benchmark():
    lab = ComputerVisionLab()
    image = np.random.rand(512, 512)
    kernel = np.random.rand(3, 3)

    start_time = time.time()
    lab.convolution2d(image, kernel, stride=1, padding=1)
    print(f"Convolution time: {time.time() - start_time:.4f} seconds")

    start_time = time.time()
    lab.max_pooling2d(image, pool_size=2)
    print(f"Max pooling time: {time.time() - start_time:.4f} seconds")

if __name__ == '__main__':
    run_benchmark()
