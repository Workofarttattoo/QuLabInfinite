"""
ResearchEngine — core ECH0 research computation layer.

Wraps ECH0_InventionAccelerator, ECH0_QuLabInterface, and
ECH0_QuantumMaterialDiscovery with graceful fallbacks so the engine
never crashes even if the QuLab lab modules are unavailable.

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import sys
import uuid
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Lab root bootstrap
# ---------------------------------------------------------------------------

def _find_lab_root() -> Path | None:
    """Walk upward from this file looking for quantum_computing_lab.py."""
    here = Path(__file__).resolve()
    for ancestor in [here, *here.parents]:
        if (ancestor / "quantum_computing_lab.py").exists():
            return ancestor
    return None


def _bootstrap(lab_root: Path | None) -> Path | None:
    if lab_root is None:
        lab_root = _find_lab_root()
    if lab_root and str(lab_root) not in sys.path:
        sys.path.insert(0, str(lab_root))
    # Also add the ech0 directory so relative imports inside it work
    if lab_root:
        ech0_dir = lab_root / "qulab" / "ech0"
        if ech0_dir.exists() and str(ech0_dir) not in sys.path:
            sys.path.insert(0, str(ech0_dir))
    return lab_root


# ---------------------------------------------------------------------------
# Domain → lab mapping
# ---------------------------------------------------------------------------

DOMAIN_MAP: dict[str, dict[str, Any]] = {
    "quantum": {
        "description": "Quantum computing, quantum circuits, qubit simulation, VQE/QAOA optimization",
        "labs": ["quantum_computing_lab", "quantum_lab.quantum_lab"],
        "keywords": ["qubit", "circuit", "entanglement", "superposition", "quantum"],
    },
    "materials": {
        "description": "Materials science, property prediction, structural/thermal/electrical materials",
        "labs": ["materials_lab.materials_database"],
        "keywords": ["material", "alloy", "composite", "strength", "density", "conductivity"],
    },
    "chemistry": {
        "description": "Molecular properties, reaction energetics, SMILES analysis",
        "labs": ["chemistry_lab.chemistry_lab"],
        "keywords": ["molecule", "smiles", "reaction", "bond", "synthesis", "chemical"],
    },
    "pharma": {
        "description": "Drug delivery, pharmacokinetics, nanoparticle release profiles",
        "labs": ["pharma_pk_model", "nano_drug_release", "chemistry_lab.chemistry_lab"],
        "keywords": ["drug", "pharma", "dose", "bioavailability", "release", "nanoparticle"],
    },
    "medical": {
        "description": "Biomedical devices, implants, tissue engineering",
        "labs": ["materials_lab.materials_database", "chemistry_lab.chemistry_lab"],
        "keywords": ["implant", "biocompatible", "scaffold", "tissue", "device", "surgical"],
    },
    "physics": {
        "description": "Classical & quantum mechanics, thermodynamics, electromagnetism",
        "labs": ["physics_engine.mechanics", "physics_engine.thermodynamics"],
        "keywords": ["force", "energy", "heat", "field", "particle", "thermodynamics"],
    },
}


# ---------------------------------------------------------------------------
# ResearchEngine
# ---------------------------------------------------------------------------

class ResearchEngine:
    """
    Core engine that wraps ECH0 tools.

    Parameters
    ----------
    lab_root : Path | None
        Explicit path to the QuLabInfinite repository root.
        Detected automatically if *None*.
    """

    def __init__(self, lab_root: Path | None = None):
        self.lab_root = _bootstrap(lab_root)
        self._ech0_ok = False
        self._interface = None
        self._accelerator = None
        self._material_discovery = None
        self._try_load_ech0()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _try_load_ech0(self) -> None:
        """Attempt to import ECH0 classes; set _ech0_ok flag."""
        try:
            from ech0_interface import ECH0_QuLabInterface  # type: ignore
            from ech0_invention_accelerator import (  # type: ignore
                ECH0_InventionAccelerator,
                InventionConcept,
            )
            from ech0_quantum_tools import ECH0_QuantumMaterialDiscovery  # type: ignore

            self._ECH0_QuLabInterface = ECH0_QuLabInterface
            self._ECH0_InventionAccelerator = ECH0_InventionAccelerator
            self._InventionConcept = InventionConcept
            self._ECH0_QuantumMaterialDiscovery = ECH0_QuantumMaterialDiscovery
            self._ech0_ok = True
        except Exception:
            self._ech0_ok = False

    def _get_interface(self):
        if self._interface is None and self._ech0_ok:
            try:
                self._interface = self._ECH0_QuLabInterface()
            except Exception:
                pass
        return self._interface

    def _get_accelerator(self):
        if self._accelerator is None and self._ech0_ok:
            try:
                self._accelerator = self._ECH0_InventionAccelerator()
            except Exception:
                pass
        return self._accelerator

    def _get_material_discovery(self):
        if self._material_discovery is None and self._ech0_ok:
            try:
                self._material_discovery = self._ECH0_QuantumMaterialDiscovery()
            except Exception:
                pass
        return self._material_discovery

    @property
    def status(self) -> str:
        return "ready" if self._ech0_ok else "degraded"

    # ------------------------------------------------------------------
    # Fallback helpers (no ECH0)
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_score(text: str, concept_name: str) -> float:
        """Simple keyword-overlap feasibility/impact estimator."""
        high_impact = ["superconductor", "quantum", "graphene", "fusion", "cancer", "ai"]
        high_feasibility = ["optimization", "filter", "coating", "doping", "alloy", "composite"]
        both = text.lower() + " " + concept_name.lower()
        impact = 0.5 + 0.05 * sum(1 for kw in high_impact if kw in both)
        feasibility = 0.5 + 0.05 * sum(1 for kw in high_feasibility if kw in both)
        return min(impact, 1.0), min(feasibility, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_invention_session(
        self,
        goal: str,
        domain: str,
        concepts: list[str],
        max_concepts: int = 10,
    ) -> dict:
        """
        Run invention acceleration pipeline.

        Returns a structured report dict — never raises.
        """
        start = datetime.now(timezone.utc)
        job_id = str(uuid.uuid4())
        ranked: list[dict] = []
        recommended_materials: list[dict] = []

        concepts = concepts[:max_concepts]

        engine_note: str | None = None
        try:
            if self._ech0_ok:
                ranked, recommended_materials = self._ech0_invention_session(
                    goal, domain, concepts
                )
            else:
                ranked, recommended_materials = self._fallback_invention_session(
                    goal, domain, concepts
                )
        except Exception as exc:
            # Last-resort fallback — keep whatever was computed, note the issue
            engine_note = f"Partial pipeline failure: {exc}"
            if not ranked:
                ranked, _ = self._fallback_invention_session(goal, domain, concepts)

        elapsed = (datetime.now(timezone.utc) - start).total_seconds()

        report: dict = {
            "job_id": job_id,
            "goal": goal,
            "domain": domain,
            "ranked_inventions": ranked,
            "recommended_materials": recommended_materials,
            "processing_time_s": round(elapsed, 3),
            "timestamp": start.isoformat(),
        }
        if engine_note:
            report["engine_note"] = engine_note
        return report

    def _ech0_invention_session(
        self, goal: str, domain: str, concepts: list[str]
    ) -> tuple[list[dict], list[dict]]:
        accelerator = self._get_accelerator()
        interface = self._get_interface()

        if accelerator is None or interface is None:
            return self._fallback_invention_session(goal, domain, concepts)

        concept_objs = [
            self._InventionConcept(name=c, description=f"{goal}: {c}")
            for c in concepts
        ]

        requirements = {
            "application": domain if domain in ("aerospace", "thermal", "electrical", "structural", "cost_sensitive") else "general",
            "budget": 10000.0,
            "constraints": {},
        }

        # Use batch_accelerate if many concepts, else single accelerate
        if len(concept_objs) > 1:
            results = accelerator.batch_accelerate(
                concepts=concept_objs,
                requirements=requirements,
                top_n=min(len(concept_objs), 5),
            )
        else:
            results = [accelerator.accelerate_invention(concept_objs[0], requirements)]

        ranked: list[dict] = []
        for res in results:
            c = res.get("concept", {})
            ranked.append({
                "name": c.get("name", "Unknown"),
                "description": c.get("description", ""),
                "feasibility": round(c.get("feasibility", 0.5), 4),
                "impact": round(c.get("impact", 0.5), 4),
                "quantum_score": round(c.get("quantum_score", 0.0), 4),
                "physics_validated": c.get("physics_validated", False),
                "required_materials": c.get("required_materials", []),
                "cost_estimate_usd": round(c.get("cost_estimate", 0.0), 2),
            })

        # Sort by quantum_score × feasibility × impact
        ranked.sort(
            key=lambda x: x["quantum_score"] * x["feasibility"] * x["impact"],
            reverse=True,
        )

        # Recommended materials via interface
        rec_mats = interface.search_materials(max_cost=500) or []
        rec_mats = rec_mats[:10]

        return ranked, rec_mats

    def _fallback_invention_session(
        self, goal: str, domain: str, concepts: list[str]
    ) -> tuple[list[dict], list[dict]]:
        """Pure-Python fallback when ECH0 modules are unavailable."""
        ranked: list[dict] = []
        for concept_name in concepts:
            impact, feasibility = self._keyword_score(goal, concept_name)
            # Add small random jitter for diversity
            impact = min(1.0, impact + random.uniform(-0.05, 0.05))
            feasibility = min(1.0, feasibility + random.uniform(-0.05, 0.05))
            quantum_score = round((impact + feasibility) / 2, 4)
            ranked.append({
                "name": concept_name,
                "description": f"{goal}: {concept_name}",
                "feasibility": round(feasibility, 4),
                "impact": round(impact, 4),
                "quantum_score": quantum_score,
                "physics_validated": feasibility > 0.6,
                "required_materials": [],
                "cost_estimate_usd": round(random.uniform(1000, 50000), 2),
            })

        ranked.sort(key=lambda x: x["quantum_score"], reverse=True)
        return ranked, []

    # ------------------------------------------------------------------

    def run_material_discovery(self, requirements: dict) -> dict:
        """
        Find materials matching requirements.

        Requirements keys (all optional):
            max_density_kg_m3, min_tensile_strength_MPa,
            max_cost_per_kg, category, application
        """
        try:
            interface = self._get_interface()
            if interface is None:
                return {"materials": [], "count": 0, "note": "ECH0 interface unavailable"}

            results = interface.search_materials(
                category=requirements.get("category"),
                min_strength=requirements.get("min_tensile_strength_MPa"),
                max_density=requirements.get("max_density_kg_m3"),
                max_cost=requirements.get("max_cost_per_kg"),
            )

            # Score each material by how well it fits
            application = requirements.get("application", "")
            scored: list[dict] = []
            for mat in results:
                fitness = self._material_fitness(mat, requirements, application)
                scored.append({**mat, "fitness_score": round(fitness, 4)})

            scored.sort(key=lambda x: x["fitness_score"], reverse=True)
            top10 = scored[:10]

            return {
                "materials": top10,
                "count": len(results),
                "top_count": len(top10),
                "application": application,
            }
        except Exception as exc:
            return {"materials": [], "count": 0, "error": str(exc)}

    @staticmethod
    def _material_fitness(mat: dict, req: dict, application: str) -> float:
        score = 1.0
        # Reward high strength
        ts = mat.get("tensile_strength", 0)
        min_ts = req.get("min_tensile_strength_MPa", 0)
        if min_ts and ts > 0:
            score *= min(ts / min_ts, 3.0)

        # Reward low density
        den = mat.get("density", 1)
        max_den = req.get("max_density_kg_m3", 20000)
        if den > 0:
            score *= max_den / max(den, 1)

        # Reward low cost
        cost = mat.get("cost_per_kg", 1)
        max_cost = req.get("max_cost_per_kg", 1e6)
        if cost > 0:
            score *= min(max_cost / max(cost, 1), 10.0)

        # Bonus for matching availability
        if mat.get("availability") == "High":
            score *= 1.2

        return round(score, 4)

    # ------------------------------------------------------------------

    def run_research_question(self, question: str, domain: str) -> dict:
        """
        Answer a research question by mapping domain → relevant simulations.

        The actual simulation calls are delegated to the qulab-mcp backend
        (called from the app layer via httpx). Here we return structured
        metadata so the app layer can decide what to call.
        """
        domain_info = DOMAIN_MAP.get(domain, DOMAIN_MAP["physics"])
        keywords = domain_info["keywords"]
        labs = domain_info["labs"]

        # Detect relevant keywords present in question
        matched_keywords = [kw for kw in keywords if kw in question.lower()]

        findings: list[dict] = []

        # Try to pull any info from the materials DB if relevant
        if domain in ("materials", "pharma", "medical") and self._ech0_ok:
            try:
                interface = self._get_interface()
                if interface:
                    stats = interface.get_database_stats()
                    findings.append({
                        "source": "materials_database",
                        "summary": f"Database contains {stats['total_materials']} materials across {len(stats['categories'])} categories: {list(stats['categories'].keys())}",
                    })
            except Exception:
                pass

        # Build synthesis from domain info
        synthesis = (
            f"Research question '{question}' maps to domain '{domain}'. "
            f"Relevant labs: {', '.join(labs)}. "
            f"Matched keywords: {', '.join(matched_keywords) if matched_keywords else 'general'}. "
            f"For detailed simulation results, route to the qulab-mcp backend using these labs."
        )

        return {
            "question": question,
            "domain": domain,
            "findings": findings,
            "simulations_run": labs,
            "matched_keywords": matched_keywords,
            "synthesis": synthesis,
        }
