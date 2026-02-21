"""
Virtual Product Development Lab API
===================================

FastAPI REST API for the Virtual Product Development Laboratory.
Provides programmatic access to product development, BOM management,
collaboration, and optimization features.

Copyright (c) Joshua Hendricks Cole (DBA: Corporation of Light)
PATENT PENDING - All Rights Reserved
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn

from .vpd_lab import VirtualProductLab
from .product_definition import DesignDiscipline, ComponentType
from .bom_manager import BOMType
from .collaboration import StakeholderRole, TaskPriority, ChangeType


# =============================================================================
# API Models
# =============================================================================

class ProductCreate(BaseModel):
    name: str
    description: str = ""
    product_family: str = ""


class ComponentCreate(BaseModel):
    name: str
    component_type: str = "PART"
    discipline: str = "MECHANICAL"
    parent_id: Optional[str] = None
    description: str = ""
    mass: float = 0.0
    unit_cost: float = 0.0


class VariantCreate(BaseModel):
    name: str
    description: str = ""


class StakeholderCreate(BaseModel):
    name: str
    email: str
    role: str = "DESIGN_ENGINEER"
    organization: str = ""


class TaskCreate(BaseModel):
    title: str
    description: str
    assigned_to: List[str]
    priority: str = "MEDIUM"


class OptimizationSetup(BaseModel):
    variables: List[Dict[str, Any]]
    objectives: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]] = []


class ExperimentRequest(BaseModel):
    type: str
    parameters: Dict[str, Any] = {}


# =============================================================================
# API Application
# =============================================================================

app = FastAPI(
    title="Virtual Product Development Lab API",
    description=(
        "RESTful API for multi-discipline virtual product development. "
        "Provides product definition, BOM management, collaboration, and optimization."
    ),
    version="1.0.0",
    contact={
        "name": "Joshua Hendricks Cole",
        "url": "https://github.com/QuLabInfinite",
    },
    license_info={
        "name": "Proprietary - PATENT PENDING",
    }
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lab instance
lab = VirtualProductLab({
    'project_name': 'VPD API',
    'organization': 'QuLab'
})


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/")
async def root():
    """API root - returns basic info."""
    return {
        "name": "Virtual Product Development Lab API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "status": "/status",
            "capabilities": "/capabilities"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "vpd-lab-api"}


@app.get("/status")
async def status():
    """Get lab status."""
    return lab.get_status()


@app.get("/capabilities")
async def capabilities():
    """Get lab capabilities."""
    return lab.get_capabilities()


# =============================================================================
# Product Management Endpoints
# =============================================================================

@app.post("/products")
async def create_product(product: ProductCreate):
    """Create a new product definition."""
    result = lab.create_product(
        name=product.name,
        description=product.description,
        product_family=product.product_family
    )
    return {"success": True, "product_id": result.id, "product": result.to_dict()}


@app.get("/products")
async def list_products():
    """List all products."""
    return {
        "products": [p.to_dict() for p in lab.products.values()]
    }


@app.get("/products/{product_id}")
async def get_product(product_id: str):
    """Get a specific product."""
    product = lab.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product.to_dict()


@app.post("/products/{product_id}/components")
async def add_component(product_id: str, component: ComponentCreate):
    """Add a component to a product."""
    try:
        comp_type = ComponentType[component.component_type]
        discipline = DesignDiscipline[component.discipline]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid enum value: {e}")

    result = lab.add_component(
        product_id=product_id,
        name=component.name,
        component_type=comp_type,
        discipline=discipline,
        parent_id=component.parent_id,
        description=component.description,
        mass=component.mass,
        unit_cost=component.unit_cost
    )

    if not result:
        raise HTTPException(status_code=404, detail="Product not found")

    return {"success": True, "component_id": result.id, "component": result.to_dict()}


@app.get("/products/{product_id}/components")
async def list_components(product_id: str):
    """List components of a product."""
    product = lab.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return {
        "components": [c.to_dict() for c in product.components.values()]
    }


@app.post("/products/{product_id}/variants")
async def create_variant(product_id: str, variant: VariantCreate):
    """Create a product variant."""
    result = lab.create_variant(
        product_id=product_id,
        name=variant.name,
        description=variant.description
    )

    if not result:
        raise HTTPException(status_code=404, detail="Product not found")

    return {"success": True, "variant_id": result.id, "variant": result.to_dict()}


# =============================================================================
# BOM Management Endpoints
# =============================================================================

@app.post("/products/{product_id}/bom")
async def generate_bom(product_id: str, bom_type: str = "ENGINEERING",
                       variant_id: Optional[str] = None):
    """Generate Bill of Materials."""
    try:
        bt = BOMType[bom_type]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid BOM type: {bom_type}")

    result = lab.generate_bom(product_id, variant_id, bt)

    if not result.get('success'):
        raise HTTPException(status_code=404, detail=result.get('error', 'Unknown error'))

    return result


@app.get("/products/{product_id}/bom/csv")
async def export_bom_csv(product_id: str):
    """Export BOM as CSV."""
    product = lab.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    lab.generate_bom(product_id)
    return {"csv": lab.get_bom_csv()}


@app.get("/products/{product_id}/bom/json")
async def export_bom_json(product_id: str):
    """Export BOM as JSON."""
    product = lab.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    lab.generate_bom(product_id)
    return {"json": lab.get_bom_json()}


# =============================================================================
# Collaboration Endpoints
# =============================================================================

@app.post("/collaboration/stakeholders")
async def add_stakeholder(stakeholder: StakeholderCreate):
    """Add a stakeholder."""
    try:
        role = StakeholderRole[stakeholder.role]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {stakeholder.role}")

    result = lab.add_stakeholder(
        name=stakeholder.name,
        email=stakeholder.email,
        role=role,
        organization=stakeholder.organization
    )
    return {"success": True, "stakeholder_id": result.id, "stakeholder": result.to_dict()}


@app.get("/collaboration/stakeholders")
async def list_stakeholders():
    """List all stakeholders."""
    return {
        "stakeholders": [s.to_dict() for s in lab.collaboration_hub.stakeholders.values()]
    }


@app.post("/collaboration/tasks")
async def create_task(task: TaskCreate):
    """Create a collaboration task."""
    try:
        priority = TaskPriority[task.priority]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid priority: {task.priority}")

    result = lab.create_task(
        title=task.title,
        description=task.description,
        assigned_to=task.assigned_to,
        priority=priority
    )
    return {"success": True, "task_id": result.id, "task": result.to_dict()}


@app.get("/collaboration/tasks")
async def list_tasks():
    """List all tasks."""
    return {
        "tasks": [t.to_dict() for t in lab.collaboration_hub.tasks.values()]
    }


@app.get("/collaboration/dashboard")
async def get_dashboard():
    """Get collaboration dashboard."""
    return lab.collaboration_hub.get_project_dashboard()


# =============================================================================
# Optimization Endpoints
# =============================================================================

@app.post("/optimization/setup")
async def setup_optimization(setup: OptimizationSetup):
    """Setup optimization problem."""
    lab.setup_optimization(
        variables=setup.variables,
        objectives=setup.objectives,
        constraints=setup.constraints
    )
    return {"success": True, "status": lab.optimizer.get_status()}


@app.post("/optimization/run")
async def run_optimization(method: str = "genetic"):
    """Run optimization."""
    result = lab.run_optimization(method)
    return result


@app.post("/optimization/trade-off")
async def trade_off_study(objective_x: str, objective_y: str, num_points: int = 20):
    """Run trade-off study."""
    result = lab.run_trade_off_study(objective_x, objective_y, num_points)
    return result


@app.post("/optimization/sensitivity")
async def sensitivity_analysis(solution: Dict[str, float]):
    """Run sensitivity analysis."""
    result = lab.run_sensitivity_analysis(solution)
    return {"success": True, "sensitivities": result}


@app.get("/optimization/status")
async def optimization_status():
    """Get optimization status."""
    return lab.optimizer.get_status()


# =============================================================================
# Experiment Endpoint
# =============================================================================

@app.post("/experiment")
async def run_experiment(request: ExperimentRequest):
    """Run a VPD lab experiment."""
    spec = {"type": request.type, **request.parameters}
    result = lab.run_experiment(spec)
    return result


# =============================================================================
# Demo Endpoint
# =============================================================================

@app.post("/demo")
async def run_demo():
    """Run the VPD lab demonstration."""
    result = lab.run_demo()
    return result


# =============================================================================
# Server Entry Point
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8020):
    """Run the API server."""
    print(f"\n{'='*60}")
    print("  Virtual Product Development Lab API")
    print(f"{'='*60}")
    print(f"\n  Server: http://{host}:{port}")
    print(f"  Docs:   http://{host}:{port}/docs")
    print(f"  Health: http://{host}:{port}/health\n")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
