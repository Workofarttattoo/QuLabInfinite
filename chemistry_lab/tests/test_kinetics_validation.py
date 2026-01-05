"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

test_kinetics_validation - Part of Chemistry Lab
"""

"""
Kinetics validation tests ensure simulated reaction rates align with benchmarks.
"""

import os
import sys
import unittest


# Ensure repository root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from chemistry_lab.validation.kinetics_validation import run_kinetics_validation  # noqa: E402


class TestKineticsValidation(unittest.TestCase):
    """Validate reaction kinetics against benchmark data."""

    def test_benchmarks_within_tolerance(self):
        results = run_kinetics_validation()
        failures = [result for result in results if not result["passed"]]

        debug_info = "\n".join(
            f"{res['reaction']}: rate={res['rate_constant']:.3e}, "
            f"expected={res['expected_rate_constant']:.3e}, "
            f"rel_error={res['relative_error']:.3f} (tol={res['tolerance']})"
            for res in results
        )
        self.assertFalse(
            failures,
            msg=f"Kinetics validation exceeded tolerance for {len(failures)} reactions:\n{debug_info}",
        )


if __name__ == "__main__":
    unittest.main()
