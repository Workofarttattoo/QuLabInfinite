"""
Enhanced BaseLab ABC with auto-registration decorator.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabMetadata:
    """Metadata for a registered lab."""

    name: str
    category: str
    description: str = ""
    version: str = "1.0.0"
    author: str = "Joshua Hendricks Cole"
    tags: tuple = ()
    is_medical: bool = False


# Global registry of lab classes populated by @register_lab
_LAB_REGISTRY: Dict[str, type] = {}
_LAB_METADATA: Dict[str, LabMetadata] = {}


def register_lab(
    name: str,
    category: str,
    description: str = "",
    version: str = "1.0.0",
    tags: tuple = (),
    is_medical: bool = False,
):
    """
    Class decorator that registers a lab in the global registry.

    Usage::

        @register_lab(
            name="oncology",
            category="medical",
            description="Production-grade oncology simulation lab",
            is_medical=True,
        )
        class OncologyLab(BaseLab):
            ...
    """

    def decorator(cls):
        meta = LabMetadata(
            name=name,
            category=category,
            description=description,
            version=version,
            author="Joshua Hendricks Cole",
            tags=tags,
            is_medical=is_medical,
        )
        _LAB_REGISTRY[name] = cls
        _LAB_METADATA[name] = meta
        cls._lab_metadata = meta
        logger.debug("Registered lab: %s (%s)", name, category)
        return cls

    return decorator


def get_registered_labs() -> Dict[str, type]:
    """Return all registered lab classes."""
    return dict(_LAB_REGISTRY)


def get_lab_metadata() -> Dict[str, LabMetadata]:
    """Return metadata for all registered labs."""
    return dict(_LAB_METADATA)


def get_labs_by_category(category: str) -> Dict[str, type]:
    """Return labs filtered by category."""
    return {
        name: cls for name, cls in _LAB_REGISTRY.items()
        if _LAB_METADATA.get(name, LabMetadata(name=name, category="")).category == category
    }


class BaseLab(ABC):
    """
    Abstract base class for all QuLabInfinite laboratories.

    Every lab must implement:
      - run_experiment(spec) -> results dict
      - get_status() -> status dict

    Optional overrides:
      - get_capabilities() -> capabilities dict
      - validate_experiment(spec) -> bool
      - cleanup()
    """

    _lab_metadata: ClassVar[Optional[LabMetadata]] = None

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self._experiment_count = 0
        self._created_at = time.time()
        self._last_experiment_at: Optional[float] = None
        logger.info("Initializing %s...", self.__class__.__name__)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run_experiment(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Run an experiment and return results."""
        ...

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return current lab status."""
        ...

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def get_capabilities(self) -> Dict[str, Any]:
        """Return lab capabilities (override for richer info)."""
        meta = getattr(self, "_lab_metadata", None)
        if meta:
            return {
                "name": meta.name,
                "category": meta.category,
                "description": meta.description,
                "version": meta.version,
                "tags": list(meta.tags),
                "is_medical": meta.is_medical,
            }
        return {"name": self.__class__.__name__, "capabilities": "N/A"}

    def validate_experiment(self, experiment_spec: Dict[str, Any]) -> bool:
        """Validate experiment spec before running. Override for custom checks."""
        return True

    def cleanup(self) -> None:
        """Clean up resources. Called when the lab is unloaded."""
        pass

    # ------------------------------------------------------------------
    # Built-in tracking
    # ------------------------------------------------------------------

    def _track_experiment(self) -> None:
        self._experiment_count += 1
        self._last_experiment_at = time.time()

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._created_at

    @property
    def metadata(self) -> Optional[LabMetadata]:
        return self._lab_metadata
