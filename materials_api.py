#!/usr/bin/env python3
"""
Materials API - RESTful access to 6.6M+ materials database

Provides:
- Fast material search by properties
- Property prediction
- Material recommendations
- Integration with lab experiments
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import sqlite3
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="QuLabInfinite Materials API",
    description="Access 6.6M+ materials for infinite lab experiments",
    version="1.0.0"
)

# Database connection
DB_PATH = Path("data/materials_comprehensive.db")


def get_db():
    """Get database connection"""
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Materials database not found. Run: python ingest/sources/comprehensive_materials_builder.py"
        )

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


@app.on_event("startup")
async def startup():
    """Verify database exists on startup"""
    if not DB_PATH.exists():
        logger.warning(f"Materials database not found at {DB_PATH}")
        logger.warning("Run: python ingest/sources/comprehensive_materials_builder.py")
    else:
        logger.info(f"✓ Materials database loaded: {DB_PATH}")
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM materials")
        count = cursor.fetchone()[0]
        logger.info(f"  Total materials: {count:,}")
        conn.close()


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "ok", "service": "QuLabInfinite Materials API"}


@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    conn = get_db()
    cursor = conn.cursor()

    # Total count
    cursor.execute("SELECT COUNT(*) FROM materials")
    total = cursor.fetchone()[0]

    # By category
    cursor.execute("""
    SELECT category, COUNT(*) as count FROM materials
    WHERE category != '' GROUP BY category
    """)
    by_category = {row[0]: row[1] for row in cursor.fetchall()}

    # Property ranges
    cursor.execute("""
    SELECT
        COUNT(CASE WHEN density > 0 THEN 1 END) as has_density,
        COUNT(CASE WHEN band_gap > 0 THEN 1 END) as has_band_gap,
        COUNT(CASE WHEN formation_energy != 0 THEN 1 END) as has_formation_energy,
        COUNT(CASE WHEN cost_per_kg > 0 THEN 1 END) as has_cost,
        AVG(CASE WHEN density > 0 THEN density END) as avg_density,
        MAX(CASE WHEN band_gap > 0 THEN band_gap END) as max_band_gap
    FROM materials
    """)
    properties = cursor.fetchone()

    conn.close()

    return {
        "total_materials": total,
        "by_category": by_category,
        "properties_coverage": {
            "density": properties[0],
            "band_gap": properties[1],
            "formation_energy": properties[2],
            "cost": properties[3]
        },
        "property_ranges": {
            "avg_density_g_cm3": round(properties[4], 2) if properties[4] else 0,
            "max_band_gap_ev": round(properties[5], 2) if properties[5] else 0
        }
    }


@app.get("/search")
async def search(
    formula: Optional[str] = Query(None, description="Chemical formula (partial match)"),
    category: Optional[str] = Query(None, description="Material category (metal, ceramic, polymer, etc)"),
    min_density: Optional[float] = Query(None, description="Minimum density (g/cm³)"),
    max_density: Optional[float] = Query(None, description="Maximum density (g/cm³)"),
    min_band_gap: Optional[float] = Query(None, description="Minimum band gap (eV)"),
    max_band_gap: Optional[float] = Query(None, description="Maximum band gap (eV)"),
    min_cost: Optional[float] = Query(None, description="Minimum cost ($/kg)"),
    max_cost: Optional[float] = Query(None, description="Maximum cost ($/kg)"),
    min_melting_point: Optional[float] = Query(None, description="Minimum melting point (K)"),
    max_melting_point: Optional[float] = Query(None, description="Maximum melting point (K)"),
    limit: int = Query(100, ge=1, le=10000, description="Max results to return")
):
    """Search materials by properties"""
    conn = get_db()
    cursor = conn.cursor()

    query = "SELECT * FROM materials WHERE 1=1"
    params = []

    # Build query dynamically
    if formula:
        query += " AND formula LIKE ?"
        params.append(f"%{formula}%")

    if category:
        query += " AND category = ?"
        params.append(category)

    if min_density is not None:
        query += " AND density >= ? AND density > 0"
        params.append(min_density)

    if max_density is not None:
        query += " AND density <= ? AND density > 0"
        params.append(max_density)

    if min_band_gap is not None:
        query += " AND band_gap >= ? AND band_gap > 0"
        params.append(min_band_gap)

    if max_band_gap is not None:
        query += " AND band_gap <= ? AND band_gap > 0"
        params.append(max_band_gap)

    if min_cost is not None:
        query += " AND cost_per_kg >= ? AND cost_per_kg > 0"
        params.append(min_cost)

    if max_cost is not None:
        query += " AND cost_per_kg <= ? AND cost_per_kg > 0"
        params.append(max_cost)

    if min_melting_point is not None:
        query += " AND melting_point >= ? AND melting_point > 0"
        params.append(min_melting_point)

    if max_melting_point is not None:
        query += " AND melting_point <= ? AND melting_point > 0"
        params.append(max_melting_point)

    query += f" LIMIT {limit}"

    cursor.execute(query, params)

    results = []
    for row in cursor.fetchall():
        material = dict(row)
        # Parse JSON fields
        material['sources'] = json.loads(material['sources'] or '[]')
        material['element_composition'] = json.loads(material['element_composition'] or '{}')
        material['data_sources'] = json.loads(material['data_sources'] or '{}')
        results.append(material)

    conn.close()

    return {
        "query": {
            "formula": formula,
            "category": category,
            "density": {"min": min_density, "max": max_density},
            "band_gap": {"min": min_band_gap, "max": max_band_gap}
        },
        "results": results,
        "count": len(results)
    }


@app.get("/material/{material_id}")
async def get_material(material_id: str):
    """Get specific material by ID"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM materials WHERE material_id = ?", (material_id,))
    row = cursor.fetchone()

    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Material {material_id} not found")

    material = dict(row)
    material['sources'] = json.loads(material['sources'] or '[]')
    material['element_composition'] = json.loads(material['element_composition'] or '{}')
    material['data_sources'] = json.loads(material['data_sources'] or '{}')

    return material


@app.get("/categories")
async def get_categories():
    """Get all available material categories"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("""
    SELECT DISTINCT category FROM materials
    WHERE category != '' ORDER BY category
    """)

    categories = [row[0] for row in cursor.fetchall()]
    conn.close()

    return {"categories": categories}


@app.get("/recommend")
async def recommend(
    use_case: str = Query(..., description="Use case: structural, thermal, electrical, optical"),
    constraint_density_max: Optional[float] = Query(None, description="Max density constraint"),
    constraint_cost_max: Optional[float] = Query(None, description="Max cost constraint"),
    limit: int = Query(10, ge=1, le=100)
):
    """Get material recommendations for specific use cases"""
    conn = get_db()
    cursor = conn.cursor()

    # Different queries for different use cases
    use_case_queries = {
        "structural": {
            "order_by": "tensile_strength DESC, density ASC",
            "category_filter": "metal"
        },
        "thermal": {
            "order_by": "thermal_conductivity DESC",
            "category_filter": None
        },
        "electrical": {
            "order_by": "electrical_conductivity DESC",
            "category_filter": "metal"
        },
        "optical": {
            "order_by": "band_gap DESC",
            "category_filter": None
        },
        "lightweight": {
            "order_by": "tensile_strength DESC, density ASC",
            "category_filter": None
        }
    }

    if use_case not in use_case_queries:
        raise HTTPException(status_code=400, detail=f"Unknown use case: {use_case}")

    query_config = use_case_queries[use_case]
    order_by = query_config["order_by"]

    query = "SELECT * FROM materials WHERE 1=1"
    params = []

    if query_config["category_filter"]:
        query += " AND category = ?"
        params.append(query_config["category_filter"])

    if constraint_density_max:
        query += " AND density > 0 AND density <= ?"
        params.append(constraint_density_max)

    if constraint_cost_max:
        query += " AND cost_per_kg > 0 AND cost_per_kg <= ?"
        params.append(constraint_cost_max)

    query += f" ORDER BY {order_by} LIMIT {limit}"

    cursor.execute(query, params)

    results = []
    for row in cursor.fetchall():
        material = dict(row)
        material['sources'] = json.loads(material['sources'] or '[]')
        results.append(material)

    conn.close()

    return {
        "use_case": use_case,
        "recommendations": results,
        "count": len(results)
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Materials API...")
    logger.info("http://localhost:8888/docs")

    uvicorn.run(app, host="0.0.0.0", port=8888)
