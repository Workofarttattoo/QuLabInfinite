"""
Wrapper utilities for standalone labs that don't inherit BaseLab.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from qulab.core.base_lab import BaseLab, LabMetadata, register_lab


def wrap_standalone_lab(
    name: str,
    instance: Any,
    category: str = "standalone",
    description: str = "",
    is_medical: bool = False,
) -> Type[BaseLab]:
    """
    Create a BaseLab subclass that wraps a standalone lab instance.

    This lets legacy labs (standalone FastAPI apps, dataclass-based labs, etc.)
    participate in the unified registry without modifying their original code.
    """

    class WrappedLab(BaseLab):
        _lab_metadata = LabMetadata(
            name=name,
            category=category,
            description=description,
            is_medical=is_medical,
        )

        def __init__(self, config: Dict[str, Any] | None = None):
            super().__init__(config)
            self._inner = instance

        def run_experiment(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
            self._track_experiment()
            # Try common method names
            for method_name in [
                "run_experiment",
                "simulate",
                "run",
                "predict",
                "analyze",
                "calculate",
            ]:
                method = getattr(self._inner, method_name, None)
                if callable(method):
                    return method(**experiment_spec) if experiment_spec else method()

            return {
                "status": "error",
                "message": f"No callable experiment method found on {type(self._inner).__name__}",
            }

        def get_status(self) -> Dict[str, Any]:
            status_method = getattr(self._inner, "get_status", None)
            if callable(status_method):
                return status_method()
            return {
                "lab": name,
                "category": category,
                "status": "available",
                "inner_type": type(self._inner).__name__,
            }

        def get_capabilities(self) -> Dict[str, Any]:
            caps = getattr(self._inner, "get_capabilities", None)
            if callable(caps):
                return caps()
            return super().get_capabilities()

    WrappedLab.__name__ = f"Wrapped_{name}"
    WrappedLab.__qualname__ = f"Wrapped_{name}"
    return WrappedLab
