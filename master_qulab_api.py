"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Master QuLab API - Unified FastAPI Gateway
Aggregates all 20+ quantum lab services with authentication, rate limiting, and monitoring
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from fastapi import FastAPI, HTTPException, Depends, Header, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Provide safe defaults before attempting heavy imports
genetic_api = cancer_metabolic_api = drug_interaction_api = None
immune_api = neurotransmitter_api = microbiome_api = None
metabolic_api = stem_cell_api = medical_safety_api = None
QuantumLab = MaterialsLab = ChemistryLab = None
FrequencyLab = OncologyLab = ProteinFoldingLab = None
CardiovascularPlaqueLab = TumorEvolutionLab = None
LABS_AVAILABLE = False

# Import all lab APIs
try:
    from genetic_variant_analyzer_api import app as genetic_api
    from cancer_metabolic_optimizer_api import app as cancer_metabolic_api
    from drug_interaction_network_api import app as drug_interaction_api
    from immune_response_simulator_api import app as immune_api
    from neurotransmitter_optimizer_api import app as neurotransmitter_api
    from microbiome_optimizer_api import app as microbiome_api
    from metabolic_syndrome_reversal_api import app as metabolic_api
    from stem_cell_predictor_api import app as stem_cell_api
    from chemistry_lab.medical_safety_api import app as medical_safety_api

    from quantum_lab.quantum_lab import QuantumLab
    from materials_lab.materials_lab import MaterialsLab
    from chemistry_lab.chemistry_lab import ChemistryLab
    from frequency_lab.frequency_lab import FrequencyLab
    from oncology_lab.oncology_lab import OncologyLab
    from protein_folding_lab_lab import ProteinFoldingLab
    from cardiovascular_plaque_lab import CardiovascularPlaqueLab
    from realistic_tumor_lab import TumorEvolutionLab

    LABS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some lab imports failed: {e}")

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple in-memory auth tokens (production should use database)
VALID_API_KEYS = {
    "qulab_master_key_2025": {"name": "Master Admin", "tier": "enterprise"},
    "qulab_demo_key": {"name": "Demo User", "tier": "demo"},
}

# Rate limiting storage
rate_limit_storage = defaultdict(list)
RATE_LIMITS = {
    "enterprise": 1000,  # requests per minute
    "demo": 60,
}

# Initialize FastAPI
app = FastAPI(
    title="QuLab Master API",
    description="Unified quantum laboratory gateway - 20+ specialized labs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Models
class LabInfo(BaseModel):
    name: str
    description: str
    status: str
    endpoints: List[str]
    version: str

class OptimizationRequest(BaseModel):
    lab_name: str = Field(..., description="Target lab name")
    parameters: Dict[str, Any] = Field(..., description="Lab-specific parameters")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    labs_status: Dict[str, str]
    system_info: Dict[str, Any]

# Authentication dependency
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify API key and return user info"""
    api_key = credentials.credentials

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return VALID_API_KEYS[api_key]

# Rate limiting dependency
async def check_rate_limit(
    request: Request,
    user_info: dict = Depends(verify_api_key)
) -> None:
    """Check rate limits based on user tier"""
    client_id = f"{user_info['name']}_{request.client.host}"
    tier = user_info["tier"]
    limit = RATE_LIMITS.get(tier, 60)

    now = time.time()
    minute_ago = now - 60

    # Clean old entries
    rate_limit_storage[client_id] = [
        ts for ts in rate_limit_storage[client_id] if ts > minute_ago
    ]

    # Check limit
    if len(rate_limit_storage[client_id]) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {limit} requests/minute for {tier} tier"
        )

    # Record request
    rate_limit_storage[client_id].append(now)

# Lab registry
LAB_REGISTRY = {
    "quantum": {
        "description": "Quantum computing and simulation",
        "class": QuantumLab if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/simulate", "/algorithms"],
    },
    "materials": {
        "description": "Materials science and discovery",
        "class": MaterialsLab if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/predict", "/simulate"],
    },
    "chemistry": {
        "description": "Chemical synthesis and analysis",
        "class": ChemistryLab if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/synthesize", "/analyze"],
    },
    "frequency": {
        "description": "Electromagnetic frequency analysis",
        "class": FrequencyLab if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/scan", "/modulate"],
    },
    "oncology": {
        "description": "Cancer biology and treatment optimization",
        "class": OncologyLab if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/simulate", "/predict"],
    },
    "protein_folding": {
        "description": "Protein structure prediction and folding",
        "class": ProteinFoldingLab if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/predict", "/fold"],
    },
    "cardiovascular": {
        "description": "Cardiovascular plaque and disease modeling",
        "class": CardiovascularPlaqueLab if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/simulate", "/analyze"],
    },
    "tumor_evolution": {
        "description": "Tumor evolution and dynamics simulation",
        "class": TumorEvolutionLab if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/simulate", "/evolve"],
    },
    "genetic_variants": {
        "description": "Genetic variant analysis and prediction",
        "api": genetic_api if LABS_AVAILABLE else None,
        "endpoints": ["/analyze", "/predict", "/variants"],
    },
    "cancer_metabolic": {
        "description": "Cancer metabolic pathway optimization",
        "api": cancer_metabolic_api if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/pathways", "/metabolites"],
    },
    "drug_interaction": {
        "description": "Drug-drug interaction network analysis",
        "api": drug_interaction_api if LABS_AVAILABLE else None,
        "endpoints": ["/analyze", "/interactions", "/network"],
    },
    "immune_response": {
        "description": "Immune system response simulation",
        "api": immune_api if LABS_AVAILABLE else None,
        "endpoints": ["/simulate", "/response", "/cells"],
    },
    "neurotransmitter": {
        "description": "Neurotransmitter optimization and balance",
        "api": neurotransmitter_api if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/balance", "/pathways"],
    },
    "microbiome": {
        "description": "Gut microbiome optimization",
        "api": microbiome_api if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/analyze", "/diversity"],
    },
    "metabolic_syndrome": {
        "description": "Metabolic syndrome reversal protocols",
        "api": metabolic_api if LABS_AVAILABLE else None,
        "endpoints": ["/optimize", "/protocols", "/biomarkers"],
    },
    "stem_cell": {
        "description": "Stem cell differentiation prediction",
        "api": stem_cell_api if LABS_AVAILABLE else None,
        "endpoints": ["/predict", "/differentiate", "/pathways"],
    },
    "medical_safety": {
        "description": "Medical safety and toxicity assessment",
        "api": medical_safety_api if LABS_AVAILABLE else None,
        "endpoints": ["/assess", "/toxicity", "/safety"],
    },
}

# Initialize lab instances
lab_instances = {}

def get_lab_instance(lab_name: str):
    """Get or create lab instance"""
    if lab_name not in lab_instances and lab_name in LAB_REGISTRY:
        lab_class = LAB_REGISTRY[lab_name].get("class")
        if lab_class:
            try:
                lab_instances[lab_name] = lab_class()
                logger.info(f"Initialized {lab_name} lab")
            except Exception as e:
                logger.error(f"Failed to initialize {lab_name}: {e}")
                return None

    return lab_instances.get(lab_name)

# Routes
@app.get("/", tags=["Root"], dependencies=[Depends(check_rate_limit)])
async def root(user_info: dict = Depends(verify_api_key)):
    """API root with quick start guide"""
    return {
        "message": "QuLab Master API - Unified Quantum Laboratory Gateway",
        "version": "1.0.0",
        "documentation": "/docs",
        "authentication": "Bearer token required (Header: Authorization: Bearer YOUR_KEY)",
        "quick_start": {
            "1_list_labs": "GET /labs",
            "2_check_health": "GET /health",
            "3_optimize": "POST /labs/{lab_name}/optimize",
        },
        "demo_key": "qulab_demo_key",
    }

@app.get("/labs", response_model=List[LabInfo], tags=["Labs"], dependencies=[Depends(check_rate_limit)])
async def list_labs(user_info: dict = Depends(verify_api_key)):
    """List all available labs with status"""
    labs = []

    for name, config in LAB_REGISTRY.items():
        lab_status = "available" if (config.get("class") or config.get("api")) else "unavailable"

        labs.append(LabInfo(
            name=name,
            description=config["description"],
            status=lab_status,
            endpoints=config["endpoints"],
            version="1.0.0",
        ))

    logger.info(f"Labs listed by {user_info['name']}")
    return labs

@app.get("/labs/{lab_name}", tags=["Labs"], dependencies=[Depends(check_rate_limit)])
async def get_lab_info(lab_name: str, user_info: dict = Depends(verify_api_key)):
    """Get detailed info about specific lab"""
    if lab_name not in LAB_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Lab '{lab_name}' not found")

    config = LAB_REGISTRY[lab_name]
    lab_instance = get_lab_instance(lab_name)

    return {
        "name": lab_name,
        "description": config["description"],
        "status": "running" if lab_instance else "not_initialized",
        "endpoints": config["endpoints"],
        "capabilities": getattr(lab_instance, "capabilities", []) if lab_instance else [],
        "version": "1.0.0",
    }

@app.post("/labs/{lab_name}/optimize", tags=["Optimization"], dependencies=[Depends(check_rate_limit)])
async def optimize_lab(
    lab_name: str,
    request: OptimizationRequest,
    user_info: dict = Depends(verify_api_key)
):
    """Route optimization request to specific lab"""
    if lab_name not in LAB_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Lab '{lab_name}' not found")

    lab_instance = get_lab_instance(lab_name)
    if not lab_instance:
        raise HTTPException(status_code=503, detail=f"Lab '{lab_name}' not available")

    try:
        # Call lab-specific optimize method
        if hasattr(lab_instance, "optimize"):
            result = await lab_instance.optimize(**request.parameters)
        elif hasattr(lab_instance, "run_optimization"):
            result = lab_instance.run_optimization(**request.parameters)
        else:
            raise HTTPException(status_code=501, detail=f"Lab '{lab_name}' does not support optimization")

        logger.info(f"Optimization on {lab_name} by {user_info['name']}")

        return {
            "lab": lab_name,
            "timestamp": datetime.utcnow().isoformat(),
            "user": user_info["name"],
            "result": result,
        }

    except Exception as e:
        logger.error(f"Optimization failed on {lab_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse, tags=["System"], dependencies=[Depends(check_rate_limit)])
async def health_check(user_info: dict = Depends(verify_api_key)):
    """Comprehensive health check across all labs"""
    labs_status = {}

    for name in LAB_REGISTRY.keys():
        lab_instance = get_lab_instance(name)
        if lab_instance:
            try:
                # Try to call health check if available
                if hasattr(lab_instance, "health_check"):
                    health = lab_instance.health_check()
                    labs_status[name] = "healthy" if health else "degraded"
                else:
                    labs_status[name] = "running"
            except Exception as e:
                labs_status[name] = f"error: {str(e)}"
        else:
            labs_status[name] = "not_initialized"

    healthy_count = sum(1 for s in labs_status.values() if s in ["healthy", "running"])
    total_count = len(LAB_REGISTRY)

    overall_status = "healthy" if healthy_count == total_count else "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        labs_status=labs_status,
        system_info={
            "healthy_labs": healthy_count,
            "total_labs": total_count,
            "uptime_seconds": time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0,
        }
    )

@app.get("/metrics", tags=["System"], dependencies=[Depends(check_rate_limit)])
async def get_metrics(user_info: dict = Depends(verify_api_key)):
    """Get API usage metrics"""
    total_requests = sum(len(reqs) for reqs in rate_limit_storage.values())

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_requests": total_requests,
        "active_clients": len(rate_limit_storage),
        "rate_limits": RATE_LIMITS,
        "user_tier": user_info["tier"],
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with logging"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler"""
    logger.error(f"Unexpected error: {exc} - {request.url}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    app.state.start_time = time.time()
    logger.info("QuLab Master API started")
    logger.info(f"Labs available: {LABS_AVAILABLE}")
    logger.info(f"Registered labs: {len(LAB_REGISTRY)}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("QuLab Master API shutting down")

if __name__ == "__main__":
    uvicorn.run(
        "master_qulab_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
