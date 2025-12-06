"""
Resource Manager - Allocate and track compute resources across labs
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import psutil
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResourceAllocation:
    """Resource allocation for a lab"""
    lab_name: str
    gpu_count: int = 0
    cpu_cores: int = 0
    memory_gb: int = 0
    allocated_at: str = ""


class ResourceManager:
    """
    Manage resource allocation across labs.

    Tracks GPU, CPU, and memory usage to prevent over-allocation.
    """

    def __init__(self):
        """Initialize resource manager"""
        # Default resource limits per lab
        self.resource_limits = {
            "cancer_optimizer": {"gpu": 1, "cpu": 4, "memory_gb": 8},
            "materials_lab": {"gpu": 2, "cpu": 8, "memory_gb": 16},
            "quantum_lab": {"gpu": 1, "cpu": 4, "memory_gb": 12},
            "medical_diagnostics": {"gpu": 0, "cpu": 2, "memory_gb": 4},
            "protein_engineering": {"gpu": 1, "cpu": 4, "memory_gb": 10},
            "toxicology_lab": {"gpu": 0, "cpu": 2, "memory_gb": 4},
            "default": {"gpu": 0, "cpu": 2, "memory_gb": 4}
        }

        # Current allocations per lab
        self.current_allocation: Dict[str, Dict[str, float]] = {}

        # System resource status
        self.system_resources = self._get_system_resources()

        logger.info(f"✓ Resource Manager initialized")
        logger.info(f"System resources: {self.system_resources}")

    def _get_system_resources(self) -> Dict[str, float]:
        """Get system resource information"""
        return {
            "total_cpu_cores": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024 ** 3),
            "available_memory_gb": psutil.virtual_memory().available / (1024 ** 3),
            "cpu_percent": psutil.cpu_percent(interval=1)
        }

    def can_allocate(self, lab_name: str) -> bool:
        """
        Check if resources are available for a lab.

        Args:
            lab_name: Name of the lab

        Returns:
            True if resources available, False otherwise
        """
        limits = self.resource_limits.get(
            lab_name,
            self.resource_limits["default"]
        )

        current = self.current_allocation.get(lab_name, {})

        # Check CPU availability
        available_cpu = (
            self.system_resources["total_cpu_cores"] -
            sum(a.get("cpu", 0) for a in self.current_allocation.values())
        )

        if current.get("cpu", 0) + limits["cpu"] > available_cpu:
            logger.warning(
                f"Insufficient CPU for {lab_name}: "
                f"need {limits['cpu']}, available {available_cpu}"
            )
            return False

        # Check memory availability
        available_memory = self.system_resources["available_memory_gb"]
        if available_memory < limits["memory_gb"]:
            logger.warning(
                f"Insufficient memory for {lab_name}: "
                f"need {limits['memory_gb']}GB, available {available_memory:.1f}GB"
            )
            return False

        return True

    def allocate(self, lab_name: str) -> Dict[str, float]:
        """
        Allocate resources to a lab.

        Args:
            lab_name: Name of the lab

        Returns:
            Dictionary of allocated resources
        """
        if not self.can_allocate(lab_name):
            logger.error(f"Cannot allocate resources for {lab_name}")
            return {}

        limits = self.resource_limits.get(
            lab_name,
            self.resource_limits["default"]
        )

        self.current_allocation[lab_name] = limits.copy()

        logger.info(
            f"Allocated to {lab_name}: "
            f"{limits['cpu']} CPU cores, "
            f"{limits['memory_gb']}GB RAM"
        )

        return limits

    def release(self, lab_name: str) -> None:
        """
        Release resources from a lab.

        Args:
            lab_name: Name of the lab
        """
        if lab_name in self.current_allocation:
            del self.current_allocation[lab_name]
            logger.info(f"Released resources from {lab_name}")

    def get_allocation(self, lab_name: str) -> Dict[str, float]:
        """
        Get current allocation for a lab.

        Args:
            lab_name: Name of the lab

        Returns:
            Current allocation dictionary
        """
        return self.current_allocation.get(lab_name, {})

    def get_status(self) -> Dict:
        """
        Get resource management status.

        Returns:
            Status dictionary
        """
        total_allocated = {
            "cpu": sum(a.get("cpu", 0) for a in self.current_allocation.values()),
            "memory_gb": sum(a.get("memory_gb", 0) for a in self.current_allocation.values())
        }

        return {
            "system_resources": self.system_resources,
            "total_allocated": total_allocated,
            "per_lab_allocation": self.current_allocation,
            "available_cpu_cores": (
                self.system_resources["total_cpu_cores"] - total_allocated["cpu"]
            ),
            "available_memory_gb": (
                self.system_resources["available_memory_gb"] - total_allocated["memory_gb"]
            )
        }
