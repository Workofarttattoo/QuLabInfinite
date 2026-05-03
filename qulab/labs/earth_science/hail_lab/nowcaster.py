import numpy as np
from typing import List, Optional

class HailNowcaster:
    """
    Monitors environmental conditions for lightning jumps and other hail precursors.
    """
    def __init__(self, flash_rate_threshold: float = 2.0):
        self.flash_rate_threshold = flash_rate_threshold # Standard deviations above mean
        self.flash_history: List[float] = []

    def detect_lightning_jump(self, current_flash_rate: float) -> bool:
        """
        Detects a 'lightning jump' defined as an increase >= threshold * std_dev.
        """
        if len(self.flash_history) < 5:
            self.flash_history.append(float(current_flash_rate))
            return False

        mean = float(np.mean(self.flash_history))
        std = float(np.std(self.flash_history))

        # Avoid division by zero
        if std == 0:
            std = 0.1

        is_jump = (float(current_flash_rate) - mean) >= (self.flash_rate_threshold * std)

        # Update history (keep last 10)
        self.flash_history.append(float(current_flash_rate))
        if len(self.flash_history) > 10:
            self.flash_history.pop(0)

        return bool(is_jump)

    def get_hail_probability(self, ship_index: float, cape: float) -> float:
        """
        Calculates probability of significant hail based on SHIP and CAPE.
        """
        # SHIP > 1.5 triggers high probability zone
        base_prob = 0.1
        if ship_index > 1.5:
            base_prob += 0.4

        # CAPE contribution
        if cape > 2000:
            base_prob += 0.3
        elif cape > 1000:
            base_prob += 0.1

        return float(min(1.0, base_prob))
