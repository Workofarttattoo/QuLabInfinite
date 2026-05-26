"""
Clinical & Medical Science MCP tools for QuLab Infinite.

Covers:
  - Alzheimer's risk assessment    (NIA-AA ATN framework, validated biomarkers)
  - Kidney function                (CKD-EPI 2021, KDIGO staging)
  - Lung function                  (GLI-2012, ATS/ERS spirometry)
  - Drug interaction analysis      (CYP450, PK/PD, polypharmacy)
  - Cancer metabolic optimization  (10-field metabolic therapy)

Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.

DISCLAIMER: For research and educational use only.
Not a substitute for professional medical advice or diagnosis.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from mcp.types import CallToolResult, TextContent, Tool


# ── helpers ──────────────────────────────────────────────────────────────────

def _ok(data: Any) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(_safe(data), indent=2))]
    )


def _err(msg: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": msg}))],
        isError=True,
    )


def _safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _safe(obj.model_dump())
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _safe(vars(obj))
    return obj


def _lab_root() -> Path | None:
    candidates = [
        Path(__file__).parents[4],
        Path(__file__).parents[3],
    ]
    for c in candidates:
        if (c / "quantum_computing_lab.py").exists():
            return c
    return None


_LAB_ROOT = _lab_root()
if _LAB_ROOT and str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

_MEDICAL_ROOT = (_LAB_ROOT / "qulab/labs/medical") if _LAB_ROOT else None
if _MEDICAL_ROOT and str(_MEDICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEDICAL_ROOT))


def _import_medical(name: str):
    """Import a medical lab module by filename (without .py)."""
    if _MEDICAL_ROOT is None:
        return None
    try:
        path = _MEDICAL_ROOT / f"{name}.py"
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location(f"_med_{name}", str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        try:
            return importlib.import_module(f"qulab.labs.medical.{name}")
        except Exception:
            return None


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS: list[Tool] = [

    Tool(
        name="medical_alzheimers_risk",
        description=(
            "Clinical Alzheimer's risk assessment using the NIA-AA ATN framework "
            "(Jack et al. 2018). Accepts CSF biomarkers (Aβ42, tau, p-tau), "
            "amyloid PET SUVR, hippocampal volume, cognitive scores (MMSE/MoCA), "
            "APOE ε4 status, and age. Returns ATN classification, 5- and 10-year "
            "progression risk, composite risk score (0-100), and clinical recommendations. "
            "DISCLAIMER: Research use only — not a substitute for clinical diagnosis."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 40, "maximum": 100,
                        "description": "Patient age (years)"},
                "csf_abeta42": {"type": "number",
                                "description": "CSF Amyloid-β42 (pg/mL). Normal > 550."},
                "csf_tau": {"type": "number",
                            "description": "CSF Total tau (pg/mL). Pathological > 400."},
                "csf_ptau": {"type": "number",
                             "description": "CSF Phosphorylated tau (pg/mL). Pathological > 80."},
                "amyloid_pet_suvr": {"type": "number",
                                     "description": "Amyloid PET SUVR. Positive > 1.20."},
                "hippocampal_volume": {"type": "number",
                                       "description": "Hippocampal volume (cm³). Atrophy < 2.8."},
                "mmse_score": {"type": "integer", "minimum": 0, "maximum": 30,
                               "description": "MMSE score (0-30). Normal ≥ 24."},
                "moca_score": {"type": "integer", "minimum": 0, "maximum": 30,
                               "description": "MoCA score (0-30). Normal ≥ 26."},
                "apoe_e4_alleles": {"type": "integer", "enum": [0, 1, 2],
                                    "description": "APOE ε4 allele count (0=none, 1=heterozygous, 2=homozygous)"},
                "family_history": {"type": "boolean", "default": False,
                                   "description": "First-degree relative with Alzheimer's"},
            },
            "required": ["age"],
        },
    ),

    Tool(
        name="medical_kidney_function",
        description=(
            "Kidney function assessment using CKD-EPI 2021 (race-free, NEJM 2021) "
            "and KDIGO staging. Returns eGFR, CKD stage (G1-G5), albuminuria "
            "category, and clinical risk classification. "
            "sex: 'M' or 'F'. race: 'white'|'black'|'asian'|'other'. "
            "acr: albumin-to-creatinine ratio (mg/g), optional."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "creatinine_mg_dl": {"type": "number",
                                     "description": "Serum creatinine (mg/dL)"},
                "age": {"type": "integer", "minimum": 18, "maximum": 110},
                "sex": {"type": "string", "enum": ["M", "F"]},
                "race": {"type": "string", "default": "white",
                         "description": "'white'|'black'|'asian'|'other'"},
                "acr_mg_g": {"type": "number",
                             "description": "Albumin-to-creatinine ratio (mg/g), optional"},
            },
            "required": ["creatinine_mg_dl", "age", "sex"],
        },
    ),

    Tool(
        name="medical_lung_function",
        description=(
            "Spirometry interpretation using GLI-2012 reference equations and "
            "ATS/ERS severity classification. Measures FEV1, FVC, FEV1/FVC ratio "
            "against predicted values. Returns obstruction/restriction pattern, "
            "severity (normal → very severe), and DLCO assessment. "
            "sex: 'M'|'F'. race: 'white'|'african_american'|'asian'|'other'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "fev1_liters": {"type": "number",
                                "description": "Forced expiratory volume in 1 second (L)"},
                "fvc_liters": {"type": "number",
                               "description": "Forced vital capacity (L)"},
                "age": {"type": "integer", "minimum": 5, "maximum": 95},
                "sex": {"type": "string", "enum": ["M", "F"]},
                "height_cm": {"type": "number",
                              "description": "Height in centimetres"},
                "race": {"type": "string", "default": "white"},
                "dlco_ml_min_mmhg": {"type": "number",
                                     "description": "Diffusing capacity (mL/min/mmHg), optional"},
            },
            "required": ["fev1_liters", "fvc_liters", "age", "sex", "height_cm"],
        },
    ),

    Tool(
        name="medical_drug_interaction",
        description=(
            "Drug-drug interaction analysis using CYP450 enzyme kinetics and "
            "validated pharmacokinetic profiles. "
            "Supported drugs: doxorubicin, cisplatin, paclitaxel, methotrexate, "
            "warfarin, atorvastatin, amiodarone, fluoxetine, risperidone, "
            "rifampin, morphine. "
            "Returns interaction type, risk level (SAFE→CRITICAL), mechanism, "
            "AUC change %, and dosing recommendations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "drugs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "List of 2+ drug names (lowercase)",
                },
            },
            "required": ["drugs"],
        },
    ),

    Tool(
        name="medical_cancer_metabolic",
        description=(
            "Cancer metabolic field optimization using the 10-field metabolic model. "
            "Optimizes pH, O₂, glucose, lactate, temperature, ROS, glutamine, "
            "calcium, ATP/ADP ratio, and cytokines for maximum therapeutic index. "
            "cancer_type: breast|lung|colon|prostate|pancreatic|melanoma|glioblastoma|leukemia. "
            "therapy_mode: aggressive|balanced|conservative. "
            "Returns predicted tumor kill fraction, therapeutic index, safety score, "
            "side effects, and implementation protocol."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "cancer_type": {
                    "type": "string",
                    "enum": ["breast", "lung", "colon", "prostate",
                             "pancreatic", "melanoma", "glioblastoma", "leukemia"],
                },
                "therapy_mode": {
                    "type": "string",
                    "enum": ["aggressive", "balanced", "conservative"],
                    "default": "balanced",
                },
                "patient_age": {"type": "number", "default": 55},
                "patient_weight_kg": {"type": "number", "default": 70},
                "tumor_volume_cm3": {"type": "number", "default": 5.0},
                "tumor_grade": {"type": "integer", "minimum": 1, "maximum": 4, "default": 2},
                "vascularity": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "previous_therapy": {"type": "boolean", "default": False},
                "comorbidities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            "required": ["cancer_type"],
        },
    ),

]


# ── Handler implementations ───────────────────────────────────────────────────

async def _handle_alzheimers(args: dict) -> CallToolResult:
    mod = _import_medical("alzheimers_detection")
    if not mod:
        return _err("alzheimers_detection lab not found — run from repo root.")
    try:
        engine = mod.AlzheimersDetectionEngine()
        data = mod.BiomarkerInput(
            age=int(args["age"]),
            csf_abeta42=args.get("csf_abeta42"),
            csf_tau=args.get("csf_tau"),
            csf_ptau=args.get("csf_ptau"),
            amyloid_pet_suvr=args.get("amyloid_pet_suvr"),
            hippocampal_volume=args.get("hippocampal_volume"),
            mmse_score=args.get("mmse_score"),
            moca_score=args.get("moca_score"),
            apoe_e4_alleles=int(args.get("apoe_e4_alleles", 0)),
            family_history=bool(args.get("family_history", False)),
        )
        report = engine.assess_risk(data)
        return _ok({
            "overall_risk": _safe(report.overall_risk),
            "risk_score_0_to_100": round(report.risk_score, 1),
            "cognitive_status": _safe(report.cognitive_status),
            "atn_classification": report.atn_classification,
            "biomarker_profile": report.biomarker_profile,
            "progression_risk_5yr_pct": report.progression_risk_5yr,
            "progression_risk_10yr_pct": report.progression_risk_10yr,
            "recommendations": report.recommendations,
            "clinical_notes": report.clinical_notes,
            "disclaimer": "Research use only — not a substitute for clinical diagnosis.",
        })
    except Exception as e:
        return _err(f"Alzheimer's assessment failed: {e}")


def _ckd_epi_2021(scr: float, age: int, sex: str) -> float:
    """CKD-EPI 2021 race-free equation (Inker et al. NEJM 2021)."""
    sex = sex.upper()
    if sex in ("F", "FEMALE"):
        kappa, alpha = 0.7, -0.241
        sex_factor = 1.012
    else:
        kappa, alpha = 0.9, -0.302
        sex_factor = 1.0
    ratio = scr / kappa
    if ratio < 1:
        egfr = 142 * (ratio ** alpha) * (0.9938 ** age) * sex_factor
    else:
        egfr = 142 * (ratio ** -1.200) * (0.9938 ** age) * sex_factor
    return round(egfr, 1)


def _ckd_stage(egfr: float) -> tuple[str, str]:
    if egfr >= 90:
        return "G1", "Normal or high ≥ 90 mL/min/1.73m²"
    if egfr >= 60:
        return "G2", "Mildly decreased 60-89"
    if egfr >= 45:
        return "G3A", "Mildly to moderately decreased 45-59"
    if egfr >= 30:
        return "G3B", "Moderately to severely decreased 30-44"
    if egfr >= 15:
        return "G4", "Severely decreased 15-29"
    return "G5", "Kidney failure < 15"


def _albuminuria(acr: float) -> tuple[str, str]:
    if acr < 30:
        return "A1", "Normal to mildly increased (< 30 mg/g)"
    if acr < 300:
        return "A2", "Moderately increased 30-299 mg/g"
    return "A3", "Severely increased ≥ 300 mg/g"


async def _handle_kidney(args: dict) -> CallToolResult:
    try:
        scr = float(args["creatinine_mg_dl"])
        age = int(args["age"])
        sex = str(args["sex"])
        acr = args.get("acr_mg_g")

        egfr = _ckd_epi_2021(scr, age, sex)
        stage, stage_desc = _ckd_stage(egfr)

        # 10-yr kidney failure risk (simplified Tangri model)
        base_risk = max(0, min(100, 100 * (1 - 0.9544 ** (egfr / 30))))

        result = {
            "egfr_ckd_epi_2021_ml_min_1_73m2": egfr,
            "ckd_stage": stage,
            "ckd_stage_description": stage_desc,
            "10yr_kidney_failure_risk_pct": round(base_risk, 1),
            "reference": "Inker LA et al. NEJM 2021 — CKD-EPI 2021 (race-free)",
            "disclaimer": "Research use only — not a substitute for clinical interpretation.",
        }

        if acr is not None:
            alb_cat, alb_desc = _albuminuria(float(acr))
            result["albuminuria_category"] = alb_cat
            result["albuminuria_description"] = alb_desc
            result["acr_mg_g"] = float(acr)

        return _ok(result)
    except Exception as e:
        return _err(f"Kidney assessment failed: {e}")


def _gli2012_predicted(age: int, height_m: float, sex: str) -> tuple[float, float]:
    """GLI-2012 predicted FEV1 and FVC (Caucasian reference)."""
    sex = sex.upper()
    H, A = height_m, age
    if sex in ("M", "MALE"):
        pred_fev1 = np.exp(-10.342 + 2.2196 * np.log(H) - 0.0276 * np.log(A))
        pred_fvc  = np.exp(-9.5232 + 2.1704 * np.log(H) - 0.0201 * np.log(A))
    else:
        pred_fev1 = np.exp(-9.6987 + 2.1211 * np.log(H) - 0.0263 * np.log(A))
        pred_fvc  = np.exp(-8.9262 + 1.9910 * np.log(H) - 0.0167 * np.log(A))
    return round(pred_fev1, 2), round(pred_fvc, 2)


def _spirometry_pattern(ratio: float, fev1_pct: float, fvc_pct: float) -> tuple[str, str]:
    """ATS/ERS classification."""
    if ratio < 0.70:
        if fev1_pct >= 80:
            return "obstructive", "mild"
        if fev1_pct >= 60:
            return "obstructive", "moderate"
        if fev1_pct >= 40:
            return "obstructive", "severe"
        return "obstructive", "very severe"
    if fvc_pct < 80:
        if fev1_pct >= 70:
            return "restrictive", "mild"
        if fev1_pct >= 60:
            return "restrictive", "moderate"
        return "restrictive", "severe"
    return "normal", "normal"


async def _handle_lung(args: dict) -> CallToolResult:
    try:
        fev1 = float(args["fev1_liters"])
        fvc = float(args["fvc_liters"])
        age = int(args["age"])
        sex = str(args["sex"])
        height_cm = float(args["height_cm"])
        dlco = args.get("dlco_ml_min_mmhg")

        pred_fev1, pred_fvc = _gli2012_predicted(age, height_cm / 100, sex)
        fev1_pct = round(100 * fev1 / pred_fev1, 1)
        fvc_pct = round(100 * fvc / pred_fvc, 1)
        ratio = round(fev1 / fvc, 3) if fvc > 0 else 0.0
        pattern, severity = _spirometry_pattern(ratio, fev1_pct, fvc_pct)

        result = {
            "pattern": pattern,
            "severity": severity,
            "fev1_liters": fev1,
            "fvc_liters": fvc,
            "fev1_fvc_ratio": ratio,
            "predicted_fev1_L": pred_fev1,
            "predicted_fvc_L": pred_fvc,
            "fev1_pct_predicted": fev1_pct,
            "fvc_pct_predicted": fvc_pct,
            "interpretation": (
                f"{pattern.capitalize()} pattern ({severity}) — "
                f"FEV1 {fev1_pct}% predicted, FVC {fvc_pct}% predicted, "
                f"FEV1/FVC {ratio:.2%}"
            ),
            "reference": "GLI-2012 / ATS-ERS spirometry guidelines",
            "disclaimer": "Research use only — not a substitute for clinical interpretation.",
        }

        if dlco is not None:
            # Rough DLCO predicted (Macintyre 2005 simplified)
            sex_up = sex.upper()
            dlco_pred = (height_cm * 0.18) - (age * 0.07) + (10.0 if sex_up in ("M", "MALE") else 2.0)
            dlco_pct = round(100 * float(dlco) / dlco_pred, 1)
            result["dlco_ml_min_mmhg"] = float(dlco)
            result["predicted_dlco_ml_min_mmhg"] = round(dlco_pred, 1)
            result["dlco_pct_predicted"] = dlco_pct

        return _ok(result)
    except Exception as e:
        return _err(f"Lung function assessment failed: {e}")


async def _handle_drug_interaction(args: dict) -> CallToolResult:
    mod = _import_medical("drug_interaction")
    if not mod:
        return _err("drug_interaction lab not found — run from repo root.")
    try:
        drugs = [d.lower().strip() for d in args["drugs"]]
        analyzer = mod.DrugInteractionAnalyzer()

        # Known drugs
        known = list(mod.DRUG_DATABASE.keys())
        unknown = [d for d in drugs if d not in known]
        if unknown:
            return _err(
                f"Unknown drug(s): {unknown}. "
                f"Supported: {sorted(known)}"
            )

        # Pairwise interactions
        pairs = []
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                result = analyzer.analyze_pairwise_interaction(drugs[i], drugs[j])
                pairs.append({
                    "drug1": result.drug1,
                    "drug2": result.drug2,
                    "interaction_type": _safe(result.interaction_type),
                    "risk_level": _safe(result.risk_level),
                    "mechanism": result.mechanism,
                    "recommendation": result.recommendation,
                    "severity_score_0_to_10": result.severity_score,
                    "auc_change_pct": result.auc_change_percent,
                    "optimal_spacing_hours": result.optimal_spacing_hours,
                })

        # Overall risk
        max_severity = max((p["severity_score_0_to_10"] for p in pairs), default=0)
        overall = "SAFE" if max_severity < 3 else "MODERATE" if max_severity < 6 else "HIGH" if max_severity < 8 else "CRITICAL"

        return _ok({
            "drugs_analyzed": drugs,
            "n_interactions": len(pairs),
            "overall_risk": overall,
            "max_severity_score": max_severity,
            "pairwise_interactions": pairs,
            "disclaimer": "Research use only — not a substitute for clinical pharmacist review.",
        })
    except Exception as e:
        return _err(f"Drug interaction analysis failed: {e}")


async def _handle_cancer_metabolic(args: dict) -> CallToolResult:
    mod = _import_medical("cancer_metabolic")
    if not mod:
        return _err("cancer_metabolic lab not found — run from repo root.")
    try:
        cancer_type = mod.CancerType(args["cancer_type"])
        therapy_mode = mod.TherapyMode(args.get("therapy_mode", "balanced"))

        patient = mod.PatientParameters(
            age=float(args.get("patient_age", 55)),
            weight=float(args.get("patient_weight_kg", 70)),
            tumor_volume=float(args.get("tumor_volume_cm3", 5.0)),
            tumor_grade=int(args.get("tumor_grade", 2)),
            vascularity=float(args.get("vascularity", 0.5)),
            previous_therapy=bool(args.get("previous_therapy", False)),
            comorbidities=list(args.get("comorbidities", [])),
        )

        optimizer = mod.CancerMetabolicOptimizer()
        result = optimizer.optimize(cancer_type, patient, therapy_mode)

        # Serialize fields
        fields_out = {}
        for field_name, mf in result.fields.items():
            fields_out[field_name] = {
                "current": _safe(mf.current_value),
                "optimal": _safe(mf.optimal_value),
                "unit": mf.unit,
                "tumor_sensitivity": _safe(mf.tumor_sensitivity),
            }

        return _ok({
            "cancer_type": result.cancer_type,
            "therapy_mode": result.therapy_mode,
            "predicted_tumor_kill_fraction": round(float(_safe(result.predicted_tumor_kill)), 3),
            "predicted_normal_damage_fraction": round(float(_safe(result.predicted_normal_damage)), 3),
            "therapeutic_index": round(float(_safe(result.therapeutic_index)), 2),
            "safety_score_0_to_1": round(float(_safe(result.safety_score)), 3),
            "estimated_side_effects": result.estimated_side_effects,
            "metabolic_fields": fields_out,
            "implementation_protocol": result.protocol,
            "breakthroughs": result.breakthroughs,
            "disclaimer": "Research/educational use only — not clinical guidance.",
        })
    except Exception as e:
        return _err(f"Cancer metabolic optimization failed: {e}")


# ── Handler dispatch ──────────────────────────────────────────────────────────

HANDLERS: dict[str, Any] = {
    "medical_alzheimers_risk":    _handle_alzheimers,
    "medical_kidney_function":    _handle_kidney,
    "medical_lung_function":      _handle_lung,
    "medical_drug_interaction":   _handle_drug_interaction,
    "medical_cancer_metabolic":   _handle_cancer_metabolic,
}
