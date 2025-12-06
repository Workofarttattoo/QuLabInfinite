"""
Result Interpolator
Advanced interpolation techniques for estimating results from nearby experiments
"""

from typing import Dict, List, Optional, Tuple
import logging
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class ResultInterpolator:
    """
    Interpolate results based on parameter differences.

    Supports:
    - Linear interpolation
    - Polynomial regression
    - Multi-dimensional scaling
    """

    def __init__(self):
        """Initialize interpolator"""
        self.numpy_available = NUMPY_AVAILABLE

    def interpolate_linear(
        self,
        known_params: Dict,
        known_result: Dict,
        target_params: Dict
    ) -> Dict:
        """
        Linear interpolation based on parameter changes.

        Args:
            known_params: Known parameter values
            known_result: Known result values
            target_params: Target parameter values

        Returns:
            Interpolated result
        """
        interpolated = {}

        for param_key, target_value in target_params.items():
            known_value = known_params.get(param_key)
            known_result_value = known_result.get(param_key)

            # Only interpolate numeric values
            if not all(
                isinstance(v, (int, float))
                for v in [target_value, known_value, known_result_value]
            ):
                # Keep original known result for non-numeric
                interpolated[param_key] = known_result_value
                continue

            # Avoid division by zero
            if known_value == 0:
                if target_value == 0:
                    interpolated[param_key] = known_result_value
                else:
                    # Scale by target magnitude
                    interpolated[param_key] = known_result_value * target_value
                continue

            # Linear scaling
            scale_factor = target_value / known_value
            interpolated[param_key] = known_result_value * scale_factor

        return interpolated

    def interpolate_polynomial(
        self,
        history: List[Tuple[Dict, Dict]],
        target_params: Dict,
        degree: int = 2
    ) -> Optional[Dict]:
        """
        Polynomial interpolation from multiple known points.

        Args:
            history: List of (parameters, result) tuples
            target_params: Target parameters
            degree: Polynomial degree

        Returns:
            Interpolated result or None
        """
        if not self.numpy_available:
            return None

        try:
            import numpy as np
            from numpy.polynomial import polynomial as P

            if len(history) < degree + 1:
                return None

            # Extract numeric parameters and results
            numeric_params = {}
            numeric_results = {}

            for params, result in history:
                for key, value in params.items():
                    if isinstance(value, (int, float)):
                        if key not in numeric_params:
                            numeric_params[key] = []
                        numeric_params[key].append(value)

                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        if key not in numeric_results:
                            numeric_results[key] = []
                        numeric_results[key].append(value)

            interpolated = {}

            # Fit polynomial for each numeric parameter
            for result_key in numeric_results:
                try:
                    # Use first numeric parameter as x
                    x_key = next(iter(numeric_params.keys()))
                    x_values = np.array(numeric_params[x_key])
                    y_values = np.array(numeric_results[result_key])

                    # Fit polynomial
                    coefficients = P.polyfit(x_values, y_values, degree)
                    poly = P.Polynomial(coefficients)

                    # Evaluate at target
                    target_x = target_params.get(x_key, np.mean(x_values))
                    interpolated[result_key] = float(poly(target_x))

                except Exception as e:
                    logger.debug(f"Polynomial fit failed for {result_key}: {e}")
                    continue

            return interpolated if interpolated else None

        except Exception as e:
            logger.warning(f"Polynomial interpolation failed: {e}")
            return None

    def estimate_uncertainty(
        self,
        known_params: Dict,
        target_params: Dict,
        known_confidences: List[float]
    ) -> float:
        """
        Estimate uncertainty in interpolated result.

        Args:
            known_params: Known parameters
            target_params: Target parameters
            known_confidences: Confidence scores of known results

        Returns:
            Estimated confidence (0-1)
        """
        # Base confidence from known data
        base_confidence = np.mean(known_confidences) if known_confidences else 0.8

        # Reduce confidence based on parameter distance
        total_distance = 0.0
        max_distance = 0.0

        for key in known_params:
            known_val = known_params[key]
            target_val = target_params.get(key)

            if (
                isinstance(known_val, (int, float)) and
                isinstance(target_val, (int, float)) and
                known_val != 0
            ):
                relative_diff = abs(target_val - known_val) / abs(known_val)
                total_distance += relative_diff
                max_distance = max(max_distance, relative_diff)

        # Apply distance penalty
        if max_distance > 0:
            # 10% penalty per 50% parameter change
            distance_penalty = min(max_distance / 5, 0.5)
            return max(0, base_confidence - distance_penalty)

        return base_confidence
