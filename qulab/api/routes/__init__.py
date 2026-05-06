"""API route sub-modules."""

from qulab.api.routes.labs import router as labs_router
from qulab.api.routes.medical import router as medical_router
from qulab.api.routes.roof_hunter import router as roof_hunter_router

__all__ = ["labs_router", "medical_router", "roof_hunter_router"]
