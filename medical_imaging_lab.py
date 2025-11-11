"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

MEDICAL IMAGING LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import k, Avogadro, g, c, h, e, pi
from typing import List

# Constants and configuration
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
VOLUME_CONTRAST = 0.5
SIGNAL_TO_NOISE_RATIO = 10


@dataclass
class MedicalImage:
    image: np.ndarray = field(init=False)
    volume_data: np.ndarray = field(default=np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT), dtype=np.float64))
    
    def __post_init__(self):
        self.generate_image()
        
    def generate_image(self) -> None:
        noise = np.random.normal(scale=1/SIGNAL_TO_NOISE_RATIO, size=(IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float64)
        self.image = (VOLUME_CONTRAST * self.volume_data + noise).clip(0, 1)

    
@dataclass
class MedicalVolumeImage(MedicalImage):
    def __post_init__(self):
        super().__post_init__()
        self.volume_data = np.random.normal(scale=0.5, size=(IMAGE_WIDTH, IMAGE_HEIGHT)).astype(np.float64)


def run_demo() -> None:
    img = MedicalVolumeImage()
    print(img.image)
    
if __name__ == '__main__':
    run_demo()
