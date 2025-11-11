"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Alzheimer's Early Detection Lab
Clinical-grade biomarker analysis using validated diagnostic criteria
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
import json

app = FastAPI(title="Alzheimer's Early Detection System", version="1.0.0")

# Clinical constants from validated research (Jack et al., 2018; NIA-AA criteria)
class AlzheimersBiomarkers:
    """Validated biomarker thresholds from NIA-AA research framework"""
    # Amyloid-β42 in CSF (pg/mL) - lower indicates pathology
    ABETA42_NORMAL_MIN = 550.0
    ABETA42_PATHOLOGICAL_MAX = 500.0

    # Total tau in CSF (pg/mL) - higher indicates neurodegeneration
    TAU_NORMAL_MAX = 300.0
    TAU_PATHOLOGICAL_MIN = 400.0

    # Phosphorylated tau (p-tau) in CSF (pg/mL)
    PTAU_NORMAL_MAX = 60.0
    PTAU_PATHOLOGICAL_MIN = 80.0

    # Amyloid PET SUVR (Standard Uptake Value Ratio)
    AMYLOID_PET_NEGATIVE_MAX = 1.10
    AMYLOID_PET_POSITIVE_MIN = 1.20

    # Hippocampal volume (cm³) age-adjusted
    HIPPOCAMPAL_VOLUME_NORMAL_MIN = 3.5
    HIPPOCAMPAL_VOLUME_ATROPHY_MAX = 2.8

    # APOE ε4 allele risk multipliers
    APOE_E4_HETEROZYGOUS_RISK = 3.0  # One copy
    APOE_E4_HOMOZYGOUS_RISK = 12.0   # Two copies

    # Cognitive scores (MMSE: Mini-Mental State Examination, 0-30)
    MMSE_NORMAL_MIN = 24
    MMSE_MCI_MAX = 23
    MMSE_MILD_DEMENTIA_MAX = 20

    # CDR (Clinical Dementia Rating): 0=normal, 0.5=questionable, 1=mild, 2=moderate, 3=severe
    # MoCA (Montreal Cognitive Assessment, 0-30)
    MOCA_NORMAL_MIN = 26
    MOCA_MCI_MAX = 25

class RiskCategory(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class CognitiveStatus(str, Enum):
    NORMAL = "normal"
    SUBJECTIVE_DECLINE = "subjective_cognitive_decline"
    MCI = "mild_cognitive_impairment"
    MILD_DEMENTIA = "mild_dementia"
    MODERATE_DEMENTIA = "moderate_dementia"

class BiomarkerInput(BaseModel):
    """Clinical biomarker measurements"""
    csf_abeta42: Optional[float] = Field(None, description="CSF Amyloid-β42 (pg/mL)", ge=100, le=2000)
    csf_tau: Optional[float] = Field(None, description="CSF Total tau (pg/mL)", ge=50, le=1500)
    csf_ptau: Optional[float] = Field(None, description="CSF Phosphorylated tau (pg/mL)", ge=10, le=200)
    amyloid_pet_suvr: Optional[float] = Field(None, description="Amyloid PET SUVR", ge=0.5, le=2.5)
    hippocampal_volume: Optional[float] = Field(None, description="Hippocampal volume (cm³)", ge=1.0, le=6.0)
    mmse_score: Optional[int] = Field(None, description="MMSE score (0-30)", ge=0, le=30)
    moca_score: Optional[int] = Field(None, description="MoCA score (0-30)", ge=0, le=30)
    age: int = Field(..., description="Patient age (years)", ge=40, le=100)
    apoe_e4_alleles: int = Field(0, description="Number of APOE ε4 alleles (0, 1, or 2)", ge=0, le=2)
    family_history: bool = Field(False, description="First-degree relative with Alzheimer's")

class AlzheimersRiskReport(BaseModel):
    """Comprehensive risk assessment report"""
    overall_risk: RiskCategory
    risk_score: float = Field(..., description="Composite risk score (0-100)")
    cognitive_status: CognitiveStatus
    biomarker_profile: Dict[str, str]
    atn_classification: str = Field(..., description="ATN framework classification (A=Amyloid, T=Tau, N=Neurodegeneration)")
    progression_risk_5yr: float = Field(..., description="5-year progression risk (%)")
    progression_risk_10yr: float = Field(..., description="10-year progression risk (%)")
    recommendations: List[str]
    clinical_notes: str

class AlzheimersDetectionEngine:
    """Clinical-grade Alzheimer's risk assessment using NIA-AA criteria"""

    def __init__(self):
        self.markers = AlzheimersBiomarkers()

    def assess_amyloid_status(self, csf_abeta42: Optional[float], amyloid_pet: Optional[float]) -> tuple[bool, float]:
        """
        Assess amyloid pathology (A in ATN framework)
        Returns: (is_positive, confidence_score)
        """
        if csf_abeta42 is not None and amyloid_pet is not None:
            # Both measurements available - highest confidence
            csf_positive = csf_abeta42 < self.markers.ABETA42_PATHOLOGICAL_MAX
            pet_positive = amyloid_pet > self.markers.AMYLOID_PET_POSITIVE_MIN

            if csf_positive and pet_positive:
                return True, 0.95
            elif csf_positive or pet_positive:
                return True, 0.75
            else:
                return False, 0.90

        elif csf_abeta42 is not None:
            if csf_abeta42 < self.markers.ABETA42_PATHOLOGICAL_MAX:
                return True, 0.85
            elif csf_abeta42 > self.markers.ABETA42_NORMAL_MIN:
                return False, 0.85
            else:
                return False, 0.60  # Intermediate range

        elif amyloid_pet is not None:
            if amyloid_pet > self.markers.AMYLOID_PET_POSITIVE_MIN:
                return True, 0.85
            elif amyloid_pet < self.markers.AMYLOID_PET_NEGATIVE_MAX:
                return False, 0.85
            else:
                return False, 0.60

        return False, 0.0  # No data

    def assess_tau_status(self, csf_tau: Optional[float], csf_ptau: Optional[float]) -> tuple[bool, float]:
        """
        Assess tau pathology (T in ATN framework)
        Returns: (is_positive, confidence_score)
        """
        if csf_tau is not None and csf_ptau is not None:
            tau_positive = csf_tau > self.markers.TAU_PATHOLOGICAL_MIN
            ptau_positive = csf_ptau > self.markers.PTAU_PATHOLOGICAL_MIN

            if tau_positive and ptau_positive:
                return True, 0.90
            elif tau_positive or ptau_positive:
                return True, 0.70
            else:
                return False, 0.85

        elif csf_tau is not None:
            if csf_tau > self.markers.TAU_PATHOLOGICAL_MIN:
                return True, 0.75
            elif csf_tau < self.markers.TAU_NORMAL_MAX:
                return False, 0.75
            else:
                return False, 0.55

        elif csf_ptau is not None:
            if csf_ptau > self.markers.PTAU_PATHOLOGICAL_MIN:
                return True, 0.80
            elif csf_ptau < self.markers.PTAU_NORMAL_MAX:
                return False, 0.80
            else:
                return False, 0.55

        return False, 0.0

    def assess_neurodegeneration(self, hippocampal_volume: Optional[float],
                                 mmse_score: Optional[int], age: int) -> tuple[bool, float]:
        """
        Assess neurodegeneration (N in ATN framework)
        Returns: (is_positive, confidence_score)
        """
        # Age-adjusted hippocampal volume (approximate 0.5% annual decline from age 55)
        age_adjusted_threshold = self.markers.HIPPOCAMPAL_VOLUME_NORMAL_MIN
        if age > 55:
            age_adjusted_threshold -= (age - 55) * 0.015

        volume_positive = False
        volume_confidence = 0.0

        if hippocampal_volume is not None:
            if hippocampal_volume < self.markers.HIPPOCAMPAL_VOLUME_ATROPHY_MAX:
                volume_positive = True
                volume_confidence = 0.85
            elif hippocampal_volume < age_adjusted_threshold:
                volume_positive = True
                volume_confidence = 0.70
            else:
                volume_confidence = 0.80

        cognitive_positive = False
        cognitive_confidence = 0.0

        if mmse_score is not None:
            if mmse_score <= self.markers.MMSE_MILD_DEMENTIA_MAX:
                cognitive_positive = True
                cognitive_confidence = 0.90
            elif mmse_score <= self.markers.MMSE_MCI_MAX:
                cognitive_positive = True
                cognitive_confidence = 0.75
            else:
                cognitive_confidence = 0.85

        # Combine evidence
        if volume_confidence > 0 and cognitive_confidence > 0:
            is_positive = volume_positive or cognitive_positive
            confidence = max(volume_confidence, cognitive_confidence)
        elif volume_confidence > 0:
            is_positive = volume_positive
            confidence = volume_confidence
        elif cognitive_confidence > 0:
            is_positive = cognitive_positive
            confidence = cognitive_confidence
        else:
            is_positive = False
            confidence = 0.0

        return is_positive, confidence

    def calculate_atn_classification(self, a_positive: bool, t_positive: bool, n_positive: bool) -> str:
        """Generate ATN classification string"""
        a_status = "A+" if a_positive else "A-"
        t_status = "T+" if t_positive else "T-"
        n_status = "N+" if n_positive else "N-"
        return f"{a_status}(T{'N' if t_positive and n_positive else n_status})"

    def calculate_genetic_risk_multiplier(self, apoe_e4_alleles: int, family_history: bool) -> float:
        """Calculate genetic risk multiplier"""
        multiplier = 1.0

        if apoe_e4_alleles == 1:
            multiplier *= self.markers.APOE_E4_HETEROZYGOUS_RISK
        elif apoe_e4_alleles == 2:
            multiplier *= self.markers.APOE_E4_HOMOZYGOUS_RISK

        if family_history:
            multiplier *= 1.8

        return multiplier

    def estimate_progression_risk(self, a_positive: bool, t_positive: bool,
                                  n_positive: bool, age: int, genetic_multiplier: float) -> tuple[float, float]:
        """
        Estimate 5-year and 10-year progression risk
        Based on longitudinal studies (Jack et al., 2018; Petersen et al., 2001)
        """
        # Base progression rates by ATN status
        if a_positive and t_positive and n_positive:
            base_5yr = 65.0  # A+T+N+ has highest progression risk
            base_10yr = 85.0
        elif a_positive and t_positive:
            base_5yr = 45.0
            base_10yr = 70.0
        elif a_positive and n_positive:
            base_5yr = 35.0
            base_10yr = 60.0
        elif a_positive:
            base_5yr = 25.0
            base_10yr = 45.0
        elif n_positive:
            base_5yr = 20.0
            base_10yr = 35.0
        else:
            base_5yr = 5.0
            base_10yr = 12.0

        # Age adjustment (risk increases with age)
        age_multiplier = 1.0 + (max(0, age - 65) * 0.02)

        # Apply multipliers
        risk_5yr = min(95.0, base_5yr * genetic_multiplier * age_multiplier)
        risk_10yr = min(98.0, base_10yr * genetic_multiplier * age_multiplier)

        return risk_5yr, risk_10yr

    def determine_cognitive_status(self, mmse_score: Optional[int],
                                   moca_score: Optional[int]) -> CognitiveStatus:
        """Determine current cognitive status"""
        if mmse_score is not None:
            if mmse_score <= self.markers.MMSE_MILD_DEMENTIA_MAX:
                return CognitiveStatus.MILD_DEMENTIA
            elif mmse_score <= self.markers.MMSE_MCI_MAX:
                return CognitiveStatus.MCI
            else:
                return CognitiveStatus.NORMAL

        if moca_score is not None:
            if moca_score <= 20:
                return CognitiveStatus.MILD_DEMENTIA
            elif moca_score <= self.markers.MOCA_MCI_MAX:
                return CognitiveStatus.MCI
            else:
                return CognitiveStatus.NORMAL

        return CognitiveStatus.NORMAL

    def calculate_composite_risk_score(self, a_positive: bool, a_confidence: float,
                                       t_positive: bool, t_confidence: float,
                                       n_positive: bool, n_confidence: float,
                                       genetic_multiplier: float, age: int) -> float:
        """Calculate composite risk score (0-100)"""
        # Weighted components
        amyloid_score = 35.0 if a_positive else 0.0
        tau_score = 30.0 if t_positive else 0.0
        neuro_score = 25.0 if n_positive else 0.0

        # Confidence weighting
        amyloid_score *= a_confidence
        tau_score *= t_confidence
        neuro_score *= n_confidence

        # Base biomarker score
        biomarker_score = amyloid_score + tau_score + neuro_score

        # Genetic and age adjustment
        genetic_factor = min(1.5, genetic_multiplier / 3.0)
        age_factor = min(1.3, 1.0 + (max(0, age - 65) * 0.015))

        final_score = min(100.0, biomarker_score * genetic_factor * age_factor)

        return round(final_score, 1)

    def generate_recommendations(self, risk_score: float, a_positive: bool,
                                t_positive: bool, n_positive: bool,
                                cognitive_status: CognitiveStatus) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []

        if risk_score >= 70:
            recommendations.append("URGENT: Immediate neurologist consultation recommended")
            recommendations.append("Consider enrollment in clinical trials for disease-modifying therapies")
        elif risk_score >= 50:
            recommendations.append("Neurologist follow-up within 3 months recommended")
            recommendations.append("Discuss early intervention strategies")
        elif risk_score >= 30:
            recommendations.append("Monitor with repeat cognitive testing in 6-12 months")
        else:
            recommendations.append("Standard cognitive screening at annual physical")

        if a_positive:
            recommendations.append("Amyloid pathology detected - discuss anti-amyloid therapies")

        if cognitive_status in [CognitiveStatus.MCI, CognitiveStatus.MILD_DEMENTIA]:
            recommendations.append("Cognitive rehabilitation and memory training programs")
            recommendations.append("Caregiver support and education resources")

        # Lifestyle interventions (always recommended)
        recommendations.extend([
            "Cardiovascular risk factor management (BP, cholesterol, diabetes control)",
            "Mediterranean-DASH diet adherence",
            "Regular aerobic exercise (150 min/week minimum)",
            "Cognitive stimulation activities",
            "Social engagement and mental health support"
        ])

        return recommendations

    def assess_risk(self, data: BiomarkerInput) -> AlzheimersRiskReport:
        """Comprehensive Alzheimer's risk assessment"""

        # ATN framework assessment
        a_positive, a_confidence = self.assess_amyloid_status(
            data.csf_abeta42, data.amyloid_pet_suvr
        )

        t_positive, t_confidence = self.assess_tau_status(
            data.csf_tau, data.csf_ptau
        )

        n_positive, n_confidence = self.assess_neurodegeneration(
            data.hippocampal_volume, data.mmse_score, data.age
        )

        # ATN classification
        atn_classification = self.calculate_atn_classification(a_positive, t_positive, n_positive)

        # Genetic risk
        genetic_multiplier = self.calculate_genetic_risk_multiplier(
            data.apoe_e4_alleles, data.family_history
        )

        # Progression risk
        risk_5yr, risk_10yr = self.estimate_progression_risk(
            a_positive, t_positive, n_positive, data.age, genetic_multiplier
        )

        # Cognitive status
        cognitive_status = self.determine_cognitive_status(data.mmse_score, data.moca_score)

        # Composite risk score
        risk_score = self.calculate_composite_risk_score(
            a_positive, a_confidence, t_positive, t_confidence,
            n_positive, n_confidence, genetic_multiplier, data.age
        )

        # Risk category
        if risk_score >= 70:
            overall_risk = RiskCategory.VERY_HIGH
        elif risk_score >= 50:
            overall_risk = RiskCategory.HIGH
        elif risk_score >= 30:
            overall_risk = RiskCategory.MODERATE
        else:
            overall_risk = RiskCategory.LOW

        # Biomarker profile
        biomarker_profile = {
            "amyloid": "POSITIVE" if a_positive else "NEGATIVE",
            "tau": "POSITIVE" if t_positive else "NEGATIVE",
            "neurodegeneration": "POSITIVE" if n_positive else "NEGATIVE"
        }

        # Recommendations
        recommendations = self.generate_recommendations(
            risk_score, a_positive, t_positive, n_positive, cognitive_status
        )

        # Clinical notes
        clinical_notes = self._generate_clinical_notes(
            data, atn_classification, risk_score, cognitive_status
        )

        return AlzheimersRiskReport(
            overall_risk=overall_risk,
            risk_score=risk_score,
            cognitive_status=cognitive_status,
            biomarker_profile=biomarker_profile,
            atn_classification=atn_classification,
            progression_risk_5yr=round(risk_5yr, 1),
            progression_risk_10yr=round(risk_10yr, 1),
            recommendations=recommendations,
            clinical_notes=clinical_notes
        )

    def _generate_clinical_notes(self, data: BiomarkerInput, atn: str,
                                 risk_score: float, status: CognitiveStatus) -> str:
        """Generate comprehensive clinical notes"""
        notes = []

        notes.append(f"Patient age {data.age} years, ATN classification: {atn}")
        notes.append(f"Cognitive status: {status.value}")
        notes.append(f"Composite risk score: {risk_score}/100")

        if data.apoe_e4_alleles > 0:
            notes.append(f"APOE ε4 carrier ({data.apoe_e4_alleles} allele{'s' if data.apoe_e4_alleles > 1 else ''})")

        if data.family_history:
            notes.append("Positive family history of Alzheimer's disease")

        notes.append(f"Assessment based on NIA-AA research framework (2018)")

        return " | ".join(notes)

# FastAPI endpoints
engine = AlzheimersDetectionEngine()

@app.post("/assess", response_model=AlzheimersRiskReport)
async def assess_alzheimers_risk(data: BiomarkerInput):
    """Comprehensive Alzheimer's risk assessment"""
    try:
        return engine.assess_risk(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/biomarker_thresholds")
async def get_biomarker_thresholds():
    """Get clinical biomarker thresholds"""
    return {
        "csf_abeta42": {
            "normal_min": AlzheimersBiomarkers.ABETA42_NORMAL_MIN,
            "pathological_max": AlzheimersBiomarkers.ABETA42_PATHOLOGICAL_MAX,
            "unit": "pg/mL"
        },
        "csf_tau": {
            "normal_max": AlzheimersBiomarkers.TAU_NORMAL_MAX,
            "pathological_min": AlzheimersBiomarkers.TAU_PATHOLOGICAL_MIN,
            "unit": "pg/mL"
        },
        "csf_ptau": {
            "normal_max": AlzheimersBiomarkers.PTAU_NORMAL_MAX,
            "pathological_min": AlzheimersBiomarkers.PTAU_PATHOLOGICAL_MIN,
            "unit": "pg/mL"
        },
        "amyloid_pet_suvr": {
            "negative_max": AlzheimersBiomarkers.AMYLOID_PET_NEGATIVE_MAX,
            "positive_min": AlzheimersBiomarkers.AMYLOID_PET_POSITIVE_MIN
        },
        "mmse": {
            "normal_min": AlzheimersBiomarkers.MMSE_NORMAL_MIN,
            "mci_max": AlzheimersBiomarkers.MMSE_MCI_MAX,
            "max_score": 30
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "operational", "system": "Alzheimer's Early Detection Lab", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    print("🧠 Alzheimer's Early Detection Lab - Clinical Grade System")
    print("📊 Using validated NIA-AA research framework criteria")
    print("🔬 ATN biomarker classification system enabled")
    uvicorn.run(app, host="0.0.0.0", port=8001)
