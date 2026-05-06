"""API route sub-modules."""

from __future__ import annotations

from typing import Any

__all__ = ["labs_router", "medical_router", "roof_hunter_router"]


def __getattr__(name: str) -> Any:
    """Load routers lazily to avoid circular imports with ``qulab.api.main``."""

    if name == "labs_router":
        from qulab.api.routes.labs import router

        return router
    if name == "medical_router":
        from qulab.api.routes.medical import router

        return router
    if name == "roof_hunter_router":
        from qulab.api.routes.roof_hunter import router

        return router
    raise AttributeError(name)
