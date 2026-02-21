"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

test_integration_snapshot - Part of Chemistry Lab
"""

"""Snapshot tests for integration experiment dataset and dashboard."""

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from chemistry_lab.tests.run_integration_experiments import (  # noqa: E402
    DASHBOARD_PATH,
    OUTPUT_PATH,
    EXPERIMENT_COUNT,
    build_dashboard,
    run_experiments,
)


def _canonical(obj):
    """Return a deterministic JSON-compatible structure for comparison."""
    return json.loads(json.dumps(obj, sort_keys=True, separators=(",", ":")))


class TestIntegrationSnapshots(unittest.TestCase):
    """Ensure generated experiment data matches the committed snapshots."""

    def setUp(self):
        self.data_path = OUTPUT_PATH
        self.dashboard_path = DASHBOARD_PATH
        if not self.data_path.exists() or not self.dashboard_path.exists():
            raise unittest.SkipTest("Integration snapshot artifacts missing; run generator first.")

    def test_experiments_snapshot(self):
        """Regenerate experiments and confirm the JSON snapshot stays stable."""
        committed = json.loads(self.data_path.read_text())
        regenerated = run_experiments(EXPERIMENT_COUNT)

        self.assertEqual(committed.get("count"), regenerated.get("count"))
        self.assertEqual(committed.get("seed"), regenerated.get("seed"))
        self.assertEqual(_canonical(committed), _canonical(regenerated))

    def test_dashboard_snapshot(self):
        """Dashboard aggregates should match the stored reference."""
        committed_dashboard = json.loads(self.dashboard_path.read_text())
        regenerated_dashboard = build_dashboard(run_experiments(EXPERIMENT_COUNT))

        self.assertEqual(_canonical(committed_dashboard), _canonical(regenerated_dashboard))


if __name__ == "__main__":
    unittest.main()
