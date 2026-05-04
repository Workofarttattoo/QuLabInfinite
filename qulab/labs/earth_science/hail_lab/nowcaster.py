import numpy as np
import os
from typing import List, Optional, Any
import logging

try:
    from hail_model.predict import HailPredictor
    HAS_ML_MODEL = True
except ImportError:
    HAS_ML_MODEL = False

logger = logging.getLogger(__name__)

class HailNowcaster:
    """
    Monitors environmental conditions for lightning jumps and other hail precursors.
    """
    def __init__(self, flash_rate_threshold: float = 2.0, model_path: Optional[str] = None):
        self.flash_rate_threshold = flash_rate_threshold # Standard deviations above mean
        self.flash_history: List[float] = []
        self.predictor: Optional[Any] = None

        if HAS_ML_MODEL:
            if model_path is None:
                # Default path
                model_path = os.path.join("models", "xgboost_hail.json")

            if os.path.exists(model_path):
                try:
                    self.predictor = HailPredictor(model_path=model_path)
                    logger.info(f"Loaded ML HailPredictor from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load HailPredictor: {e}")
            else:
                logger.warning(f"ML model path {model_path} does not exist. Falling back to heuristic.")

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

    def get_hail_probability(self, ship_index: float, cape: float, features: Optional[dict] = None) -> float:
        """
        Calculates probability of significant hail.
        Uses ML model if available and features are provided, otherwise falls back to heuristic.
        """
        if self.predictor and features:
            try:
                # Ensure CAPE and SHIP are in features if not already there
                if "cape" not in features:
                    features["cape"] = cape

                result = self.predictor.predict_full(features)
                return float(result["hail_probability"])
            except Exception as e:
                logger.error(f"ML Prediction failed, falling back: {e}")

        # Fallback Heuristic
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
