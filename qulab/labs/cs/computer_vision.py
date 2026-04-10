"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

COMPUTER VISION LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import TypeVar, Generic
import scipy.signal

T = TypeVar('T')

@dataclass
class ImageData:
    pixels: np.ndarray

@dataclass
class GrayscaleImageData(ImageData):
    @classmethod
    def from_rgb(cls, image_data: ImageData) -> 'GrayscaleImageData':
        if len(image_data.pixels.shape) != 3 or image_data.pixels.shape[2] != 3:
            raise ValueError("Input must be a 3-channel RGB array")
        
        # Convert to grayscale using luminosity method
        grayscale = np.dot(image_data.pixels[..., :3], [0.2126, 0.7152, 0.0722])
        return cls(grayscale.astype(np.float64))

@dataclass
class EdgeDetectionData(ImageData):
    @classmethod
    def from_grayscale(cls, image_data: GrayscaleImageData) -> 'EdgeDetectionData':
        # Apply Sobel filter for edge detection
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        conv_x = scipy.signal.convolve(image_data.pixels, kernel_x, mode='same', method='direct')
        conv_y = scipy.signal.convolve(image_data.pixels, kernel_y, mode='same', method='direct')

        edge_magnitude = np.sqrt(conv_x**2 + conv_y**2)
        
        return cls(edge_magnitude.astype(np.float64))

@dataclass
class ImageProcessingLab:
    image: ImageData
    
    def __post_init__(self):
        if not isinstance(self.image, ImageData):
            raise ValueError("Image must be an instance of ImageData")

    def to_grayscale(self) -> GrayscaleImageData:
        return GrayscaleImageData.from_rgb(self.image)

    def detect_edges(self) -> EdgeDetectionData:
        grayscale_data = self.to_grayscale()
        return EdgeDetectionData.from_grayscale(grayscale_data)
    
def run_demo():
    # Create a synthetic RGB image (64x64 pixels with gradient pattern)
    height, width = 64, 64
    image_array = np.zeros((height, width, 3), dtype=np.float64)

    # Create gradient pattern
    for i in range(height):
        for j in range(width):
            image_array[i, j, 0] = i / height  # Red channel
            image_array[i, j, 1] = j / width   # Green channel
            image_array[i, j, 2] = (i + j) / (height + width)  # Blue channel

    lab = ImageProcessingLab(ImageData(pixels=image_array))
    grayscale_data = lab.to_grayscale()
    edge_detection_data = lab.detect_edges()

    print("Grayscale Image shape:", grayscale_data.pixels.shape)
    print("Grayscale Image sample (first 5x5):")
    print(grayscale_data.pixels[:5, :5])

    print("\nEdge Detection Result shape:", edge_detection_data.pixels.shape)
    print("Edge Detection sample (first 5x5):")
    print(edge_detection_data.pixels[:5, :5])
    
if __name__ == '__main__':
    run_demo()