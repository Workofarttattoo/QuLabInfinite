"""
UnifiedSimulator — auto-discovers and orchestrates ALL labs.

This replaces the original 3-lab hardcoded simulator with a dynamic
registry-based system that auto-discovers every lab in qulab.labs.*.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from qulab.core.base_lab import BaseLab
from qulab.core.config import ConfigManager
from qulab.core.registry import LabRegistry

logger = logging.getLogger(__name__)


class UnifiedSimulator:
    """
    A unified interface for managing and running simulations across ALL
    QuLabInfinite laboratories.

    Auto-discovers labs via the registry system instead of hardcoding imports.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        auto_discover: bool = True,
    ):
        self.config_manager = ConfigManager(config_path)
        self.registry = LabRegistry()

        if auto_discover:
            discovered = self.registry.auto_discover("qulab.labs")
            logger.info("UnifiedSimulator loaded with %d labs", discovered)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def run_simulation(
        self, lab_name: str, experiment_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run an experiment in a specified lab."""
        lab = self.registry.get(
            lab_name, config=self.config_manager.get_lab_config(lab_name)
        )

        # Validate if the lab supports it
        if not lab.validate_experiment(experiment_spec):
            return {
                "status": "error",
                "message": f"Invalid experiment spec for lab '{lab_name}'",
            }

        return lab.run_experiment(experiment_spec)

    def get_lab_status(self, lab_name: str) -> Dict[str, Any]:
        """Get status of a specific lab."""
        lab = self.registry.get(lab_name)
        return lab.get_status()

    def list_labs(self) -> Dict[str, Dict[str, Any]]:
        """List all available labs with their capabilities."""
        result = {}
        for name in self.registry.list_labs():
            meta = self.registry.get_metadata(name)
            if meta:
                result[name] = {
                    "name": meta.name,
                    "category": meta.category,
                    "description": meta.description,
                    "version": meta.version,
                    "is_medical": meta.is_medical,
                    "tags": list(meta.tags),
                }
            else:
                result[name] = {"name": name}
        return result

    def list_categories(self) -> List[str]:
        """List all lab categories."""
        return self.registry.list_categories()

    def list_medical_labs(self) -> List[str]:
        """List all medical-grade labs."""
        return self.registry.list_medical()

    def summary(self) -> Dict[str, Any]:
        """Get a complete summary of the simulator state."""
        return {
            "simulator": "QuLabInfinite UnifiedSimulator",
            "version": self.config_manager.get("app.version", "1.0.0"),
            **self.registry.summary(),
        }


# ------------------------------------------------------------------
# CLI entry point for quick testing
# ------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sim = UnifiedSimulator()

    print("\n=== QuLabInfinite UnifiedSimulator ===")
    summary = sim.summary()
    print(f"Total labs: {summary['total_labs']}")

    for cat, labs in summary["categories"].items():
        print(f"\n  [{cat}] ({len(labs)} labs)")
        for lab in labs:
            print(f"    - {lab}")

    if summary["medical_labs"]:
        print(f"\n  Medical labs ({len(summary['medical_labs'])}):")
        for lab in summary["medical_labs"]:
            print(f"    🏥 {lab}")
