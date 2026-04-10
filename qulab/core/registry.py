"""
Lab Registry — auto-discovers and manages all labs.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from qulab.core.base_lab import BaseLab, LabMetadata, _LAB_METADATA, _LAB_REGISTRY

logger = logging.getLogger(__name__)


class LabRegistry:
    """
    Central registry for all QuLabInfinite labs.

    Supports:
      - Auto-discovery: scans qulab.labs.* for @register_lab-decorated classes
      - Manual registration of standalone labs (wraps non-BaseLab classes)
      - Filtering by category, tags, medical status
      - Lazy instantiation
    """

    def __init__(self) -> None:
        self._lab_classes: Dict[str, Type[BaseLab]] = {}
        self._lab_metadata: Dict[str, LabMetadata] = {}
        self._instances: Dict[str, BaseLab] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def auto_discover(self, package_name: str = "qulab.labs") -> int:
        """
        Recursively import all modules under `package_name` to trigger
        @register_lab decorators, then absorb them into this registry.

        Returns the number of labs discovered.
        """
        count_before = len(_LAB_REGISTRY)
        try:
            package = importlib.import_module(package_name)
        except ImportError as exc:
            logger.warning("Cannot import %s: %s", package_name, exc)
            return 0

        package_path = getattr(package, "__path__", None)
        if package_path is None:
            return 0

        for _importer, modname, _ispkg in pkgutil.walk_packages(
            package_path, prefix=package_name + "."
        ):
            try:
                importlib.import_module(modname)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping %s: %s", modname, exc)

        # Absorb globally-registered labs
        self._lab_classes.update(_LAB_REGISTRY)
        self._lab_metadata.update(_LAB_METADATA)

        discovered = len(_LAB_REGISTRY) - count_before
        logger.info(
            "Auto-discovery complete: %d new labs (%d total registered)",
            discovered,
            len(self._lab_classes),
        )
        return discovered

    # ------------------------------------------------------------------
    # Manual registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        lab_class: Type[BaseLab],
        metadata: LabMetadata | None = None,
    ) -> None:
        """Register a lab class manually."""
        self._lab_classes[name] = lab_class
        if metadata:
            self._lab_metadata[name] = metadata
        elif hasattr(lab_class, "_lab_metadata") and lab_class._lab_metadata:
            self._lab_metadata[name] = lab_class._lab_metadata

    def register_standalone(
        self,
        name: str,
        lab_instance: Any,
        category: str = "standalone",
        description: str = "",
        is_medical: bool = False,
    ) -> None:
        """
        Wrap a non-BaseLab object as a registered lab.
        Used for legacy standalone labs that don't inherit BaseLab.
        """
        from qulab.core._wrappers import wrap_standalone_lab

        wrapped_class = wrap_standalone_lab(
            name=name,
            instance=lab_instance,
            category=category,
            description=description,
            is_medical=is_medical,
        )
        meta = LabMetadata(
            name=name,
            category=category,
            description=description,
            is_medical=is_medical,
        )
        self._lab_classes[name] = wrapped_class
        self._lab_metadata[name] = meta

    # ------------------------------------------------------------------
    # Instance management
    # ------------------------------------------------------------------

    def get(self, name: str, config: Dict[str, Any] | None = None) -> BaseLab:
        """Get or create a lab instance."""
        if name not in self._instances:
            if name not in self._lab_classes:
                raise KeyError(
                    f"Unknown lab '{name}'. Available: {sorted(self._lab_classes.keys())}"
                )
            self._instances[name] = self._lab_classes[name](config=config or {})
        return self._instances[name]

    def get_metadata(self, name: str) -> Optional[LabMetadata]:
        return self._lab_metadata.get(name)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def list_labs(self) -> List[str]:
        """Return all registered lab names."""
        return sorted(self._lab_classes.keys())

    def list_by_category(self, category: str) -> List[str]:
        return sorted(
            name
            for name, meta in self._lab_metadata.items()
            if meta.category == category
        )

    def list_medical(self) -> List[str]:
        return sorted(
            name
            for name, meta in self._lab_metadata.items()
            if meta.is_medical
        )

    def list_categories(self) -> List[str]:
        return sorted({meta.category for meta in self._lab_metadata.values()})

    @property
    def count(self) -> int:
        return len(self._lab_classes)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all registered labs."""
        by_category: Dict[str, List[str]] = {}
        for name, meta in self._lab_metadata.items():
            by_category.setdefault(meta.category, []).append(name)
        return {
            "total_labs": self.count,
            "categories": {cat: sorted(labs) for cat, labs in sorted(by_category.items())},
            "medical_labs": self.list_medical(),
        }
