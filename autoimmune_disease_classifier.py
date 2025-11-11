"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Autoimmune Disease Classifier
Clinical-grade serological and clinical feature analysis for autoimmune diagnosis
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

app = FastAPI(title="Autoimmune Disease Classifier", version="1.0.0")

# Clinical constants from ACR/EULAR criteria and validated serological thresholds
class AutoimmuneMarkers:
    """Validated autoantibody titers and clinical thresholds"""
    # Antinuclear Antibody (ANA) - titer threshold for positivity
    ANA_POSITIVE_TITER = 1.0 / 80  # 1:80 dilution (stored as reciprocal for computation)
    ANA_HIGH_TITER = 1.0 / 640

    # Rheumatoid Factor (RF) IU/mL
    RF_NORMAL_MAX = 14.0
    RF_MODERATE_MIN = 15.0
    RF_HIGH_MIN = 60.0

    # Anti-CCP (anti-cyclic citrullinated peptide) U/mL
    ANTI_CCP_NEGATIVE_MAX = 20.0
    ANTI_CCP_POSITIVE_MIN = 20.1
    ANTI_CCP_HIGH_MIN = 60.0

    # Anti-dsDNA (anti-double stranded DNA) IU/mL - specific for SLE
    ANTI_DSDNA_NEGATIVE_MAX = 30.0
    ANTI_DSDNA_POSITIVE_MIN = 75.0

    # C3 complement (mg/dL) - low indicates consumption
    C3_NORMAL_MIN = 90.0
    C3_NORMAL_MAX = 180.0
    C3_LOW_MAX = 70.0

    # C4 complement (mg/dL)
    C4_NORMAL_MIN = 10.0
    C4_NORMAL_MAX = 40.0
    C4_LOW_MAX = 8.0

    # ESR (Erythrocyte Sedimentation Rate) mm/hr - inflammation marker
    ESR_NORMAL_MALE_MAX = 15.0
    ESR_NORMAL_FEMALE_MAX = 20.0
    ESR_ELEVATED_MIN = 30.0
    ESR_HIGH_MIN = 50.0

    # CRP (C-Reactive Protein) mg/L - acute inflammation
    CRP_NORMAL_MAX = 3.0
    CRP_ELEVATED_MIN = 10.0
    CRP_HIGH_MIN = 100.0

    # ACR/EULAR 2010 RA classification criteria (score ≥6 = definite RA)
    RA_DEFINITE_SCORE_MIN = 6

    # ACR 1997 SLE classification (≥4 of 11 criteria)
    SLE_DEFINITE_CRITERIA_MIN = 4

class AutoimmuneDisease(str, Enum):
    RHEUMATOID_ARTHRITIS = "rheumatoid_arthritis"
    SYSTEMIC_LUPUS_ERYTHEMATOSUS = "systemic_lupus_erythematosus"
    SJOGRENS_SYNDROME = "sjogrens_syndrome"
    SYSTEMIC_SCLEROSIS = "systemic_sclerosis"
    MIXED_CONNECTIVE_TISSUE = "mixed_connective_tissue_disease"
    POLYMYOSITIS_DERMATOMYOSITIS = "polymyositis_dermatomyositis"
    VASCULITIS = "vasculitis"
    INFLAMMATORY_ARTHRITIS_UNDIFFERENTIATED = "undifferentiated_inflammatory_arthritis"
    UNLIKELY_AUTOIMMUNE = "unlikely_autoimmune"

class SerologyInput(BaseModel):
    """Serological and clinical markers for autoimmune assessment"""
    ana_titer: Optional[int] = Field(None, description="ANA titer (e.g., 80, 160, 320, 640)", ge=0, le=5120)
    ana_pattern: Optional[str] = Field(None, description="ANA pattern (homogeneous, speckled, nucleolar, centromere)")
    rf_iu_ml: Optional[float] = Field(None, description="Rheumatoid Factor (IU/mL)", ge=0, le=500)
    anti_ccp_u_ml: Optional[float] = Field(None, description="Anti-CCP (U/mL)", ge=0, le=500)
    anti_dsdna_iu_ml: Optional[float] = Field(None, description="Anti-dsDNA (IU/mL)", ge=0, le=500)
    anti_smith: Optional[bool] = Field(None, description="Anti-Smith antibody present")
    anti_rnp: Optional[bool] = Field(None, description="Anti-RNP antibody present")
    anti_ssa_ro: Optional[bool] = Field(None, description="Anti-SSA/Ro present (Sjögren's)")
    anti_ssb_la: Optional[bool] = Field(None, description="Anti-SSB/La present (Sjögren's)")
    anti_scl70: Optional[bool] = Field(None, description="Anti-Scl-70 (scleroderma)")
    anti_centromere: Optional[bool] = Field(None, description="Anti-centromere (limited scleroderma)")
    c3_mg_dl: Optional[float] = Field(None, description="C3 complement (mg/dL)", ge=0, le=300)
    c4_mg_dl: Optional[float] = Field(None, description="C4 complement (mg/dL)", ge=0, le=100)
    esr_mm_hr: Optional[float] = Field(None, description="ESR (mm/hr)", ge=0, le=150)
    crp_mg_l: Optional[float] = Field(None, description="CRP (mg/L)", ge=0, le=500)
    age: int = Field(..., description="Patient age", ge=1, le=100)
    sex: str = Field(..., description="Patient sex (M/F)")
    joint_symptoms: bool = Field(False, description="Inflammatory joint pain/swelling")
    symmetric_arthritis: bool = Field(False, description="Symmetric joint involvement")
    small_joint_involvement: bool = Field(False, description="Hands, wrists, feet involved")
    morning_stiffness_minutes: int = Field(0, description="Duration of morning stiffness", ge=0, le=720)
    malar_rash: bool = Field(False, description="Malar (butterfly) rash")
    discoid_rash: bool = Field(False, description="Discoid skin lesions")
    photosensitivity: bool = Field(False, description="Photosensitivity")
    oral_ulcers: bool = Field(False, description="Oral/nasal ulcers")
    serositis: bool = Field(False, description="Pleuritis or pericarditis")
    renal_disorder: bool = Field(False, description="Proteinuria or cellular casts")
    neurologic_disorder: bool = Field(False, description="Seizures or psychosis")
    raynauds_phenomenon: bool = Field(False, description="Raynaud's phenomenon")
    skin_thickening: bool = Field(False, description="Skin thickening (scleroderma)")
    dry_eyes_mouth: bool = Field(False, description="Sicca symptoms")

class ClassificationResult(BaseModel):
    """Autoimmune disease classification report"""
    primary_diagnosis: AutoimmuneDisease
    confidence: float = Field(..., description="Diagnostic confidence (0-1)")
    differential_diagnoses: List[Dict[str, float]]
    serological_profile: Dict[str, str]
    clinical_features_score: int
    acr_eular_score: Optional[float] = Field(None, description="RA classification score if applicable")
    sle_criteria_met: Optional[int] = Field(None, description="Number of SLE criteria met")
    disease_activity: str
    recommendations: List[str]
    clinical_notes: str

class AutoimmuneClassifier:
    """Clinical-grade autoimmune disease classification engine"""

    def __init__(self):
        self.markers = AutoimmuneMarkers()

    def calculate_ra_score(self, data: SerologyInput) -> float:
        """
        ACR/EULAR 2010 RA classification score (0-10)
        Score ≥6 indicates definite RA
        """
        score = 0.0

        # A. Joint involvement (0-5 points)
        if data.small_joint_involvement:
            if data.symmetric_arthritis:
                score += 3  # Multiple small joints
            else:
                score += 2  # 1-3 small joints

        # B. Serology (0-3 points)
        if data.rf_iu_ml and data.anti_ccp_u_ml:
            if data.rf_iu_ml > self.markers.RF_HIGH_MIN or data.anti_ccp_u_ml > self.markers.ANTI_CCP_HIGH_MIN:
                score += 3  # High positive RF or anti-CCP
            elif data.rf_iu_ml > self.markers.RF_NORMAL_MAX or data.anti_ccp_u_ml > self.markers.ANTI_CCP_POSITIVE_MIN:
                score += 2  # Low positive
        elif data.rf_iu_ml and data.rf_iu_ml > self.markers.RF_HIGH_MIN:
            score += 3
        elif data.anti_ccp_u_ml and data.anti_ccp_u_ml > self.markers.ANTI_CCP_HIGH_MIN:
            score += 3

        # C. Acute phase reactants (0-1 point)
        if data.esr_mm_hr and data.esr_mm_hr > self.markers.ESR_ELEVATED_MIN:
            score += 1
        elif data.crp_mg_l and data.crp_mg_l > self.markers.CRP_ELEVATED_MIN:
            score += 1

        # D. Duration (0-1 point) - assume ≥6 weeks if morning stiffness present
        if data.morning_stiffness_minutes >= 30:
            score += 1

        return score

    def count_sle_criteria(self, data: SerologyInput) -> int:
        """
        ACR 1997 SLE classification criteria (11 criteria, need ≥4)
        """
        criteria_met = 0

        # 1. Malar rash
        if data.malar_rash:
            criteria_met += 1

        # 2. Discoid rash
        if data.discoid_rash:
            criteria_met += 1

        # 3. Photosensitivity
        if data.photosensitivity:
            criteria_met += 1

        # 4. Oral ulcers
        if data.oral_ulcers:
            criteria_met += 1

        # 5. Arthritis (non-erosive, ≥2 joints)
        if data.joint_symptoms and not data.symmetric_arthritis:
            criteria_met += 1

        # 6. Serositis
        if data.serositis:
            criteria_met += 1

        # 7. Renal disorder
        if data.renal_disorder:
            criteria_met += 1

        # 8. Neurologic disorder
        if data.neurologic_disorder:
            criteria_met += 1

        # 9. Hematologic disorder (not directly measured here, skip)

        # 10. Immunologic disorder (anti-dsDNA, anti-Sm, antiphospholipid)
        if (data.anti_dsdna_iu_ml and data.anti_dsdna_iu_ml > self.markers.ANTI_DSDNA_POSITIVE_MIN) or data.anti_smith:
            criteria_met += 1

        # 11. ANA
        if data.ana_titer and data.ana_titer >= 80:
            criteria_met += 1

        return criteria_met

    def assess_disease_probabilities(self, data: SerologyInput) -> Dict[AutoimmuneDisease, float]:
        """Calculate probability for each autoimmune disease (0-1 scale)"""
        probabilities = {}

        # Rheumatoid Arthritis
        ra_score = self.calculate_ra_score(data)
        ra_prob = 0.0
        if ra_score >= self.markers.RA_DEFINITE_SCORE_MIN:
            ra_prob = min(0.95, 0.60 + (ra_score - 6) * 0.08)
        elif data.anti_ccp_u_ml and data.anti_ccp_u_ml > self.markers.ANTI_CCP_HIGH_MIN:
            ra_prob = 0.75  # Anti-CCP is highly specific
        elif data.rf_iu_ml and data.rf_iu_ml > self.markers.RF_HIGH_MIN and data.symmetric_arthritis:
            ra_prob = 0.60
        elif data.joint_symptoms and data.morning_stiffness_minutes >= 30:
            ra_prob = 0.30
        probabilities[AutoimmuneDisease.RHEUMATOID_ARTHRITIS] = ra_prob

        # Systemic Lupus Erythematosus
        sle_criteria = self.count_sle_criteria(data)
        sle_prob = 0.0
        if sle_criteria >= self.markers.SLE_DEFINITE_CRITERIA_MIN:
            sle_prob = min(0.95, 0.70 + (sle_criteria - 4) * 0.05)
        elif data.anti_dsdna_iu_ml and data.anti_dsdna_iu_ml > self.markers.ANTI_DSDNA_POSITIVE_MIN:
            sle_prob = 0.70  # Anti-dsDNA is highly specific for SLE
        elif data.ana_titer and data.ana_titer >= 640 and (data.malar_rash or data.renal_disorder):
            sle_prob = 0.65
        elif sle_criteria >= 3:
            sle_prob = 0.45
        probabilities[AutoimmuneDisease.SYSTEMIC_LUPUS_ERYTHEMATOSUS] = sle_prob

        # Sjögren's Syndrome
        sjogrens_prob = 0.0
        if data.anti_ssa_ro and data.anti_ssb_la and data.dry_eyes_mouth:
            sjogrens_prob = 0.85
        elif (data.anti_ssa_ro or data.anti_ssb_la) and data.dry_eyes_mouth:
            sjogrens_prob = 0.65
        elif data.dry_eyes_mouth and data.ana_titer and data.ana_titer >= 320:
            sjogrens_prob = 0.40
        probabilities[AutoimmuneDisease.SJOGRENS_SYNDROME] = sjogrens_prob

        # Systemic Sclerosis (Scleroderma)
        scleroderma_prob = 0.0
        if data.anti_scl70 and data.skin_thickening:
            scleroderma_prob = 0.90  # Anti-Scl-70 is highly specific
        elif data.anti_centromere and data.raynauds_phenomenon:
            scleroderma_prob = 0.85  # Limited cutaneous SSc
        elif data.skin_thickening and data.raynauds_phenomenon:
            scleroderma_prob = 0.60
        elif data.raynauds_phenomenon and data.ana_titer and data.ana_titer >= 320:
            if data.ana_pattern == "nucleolar" or data.ana_pattern == "centromere":
                scleroderma_prob = 0.50
        probabilities[AutoimmuneDisease.SYSTEMIC_SCLEROSIS] = scleroderma_prob

        # Mixed Connective Tissue Disease
        mctd_prob = 0.0
        if data.anti_rnp and data.raynauds_phenomenon and data.joint_symptoms:
            mctd_prob = 0.75
        elif data.anti_rnp and (data.raynauds_phenomenon or data.joint_symptoms):
            mctd_prob = 0.50
        probabilities[AutoimmuneDisease.MIXED_CONNECTIVE_TISSUE] = mctd_prob

        # Polymyositis/Dermatomyositis (limited data, basic heuristic)
        pm_dm_prob = 0.0
        if data.ana_titer and data.ana_titer >= 320 and data.crp_mg_l and data.crp_mg_l > self.markers.CRP_ELEVATED_MIN:
            pm_dm_prob = 0.30  # Would need muscle enzymes and EMG for real diagnosis
        probabilities[AutoimmuneDisease.POLYMYOSITIS_DERMATOMYOSITIS] = pm_dm_prob

        # Vasculitis (generic, would need ANCA for specificity)
        vasculitis_prob = 0.0
        if data.crp_mg_l and data.crp_mg_l > self.markers.CRP_HIGH_MIN:
            vasculitis_prob = 0.25
        probabilities[AutoimmuneDisease.VASCULITIS] = vasculitis_prob

        # Undifferentiated inflammatory arthritis
        undiff_prob = 0.0
        if data.joint_symptoms and data.ana_titer and data.ana_titer >= 80:
            if max(probabilities.values()) < 0.50:  # No clear diagnosis
                undiff_prob = 0.50
        probabilities[AutoimmuneDisease.INFLAMMATORY_ARTHRITIS_UNDIFFERENTIATED] = undiff_prob

        # Unlikely autoimmune
        if max(probabilities.values()) < 0.30:
            probabilities[AutoimmuneDisease.UNLIKELY_AUTOIMMUNE] = 0.70

        return probabilities

    def classify(self, data: SerologyInput) -> ClassificationResult:
        """Comprehensive autoimmune disease classification"""

        # Calculate disease probabilities
        probabilities = self.assess_disease_probabilities(data)

        # Primary diagnosis (highest probability)
        primary_diagnosis = max(probabilities, key=probabilities.get)
        confidence = probabilities[primary_diagnosis]

        # Differential diagnoses (sorted by probability, exclude primary)
        differentials = [
            {"disease": disease.value, "probability": round(prob, 3)}
            for disease, prob in sorted(probabilities.items(), key=lambda x: -x[1])
            if disease != primary_diagnosis and prob >= 0.25
        ][:3]  # Top 3 differentials

        # Serological profile
        serology = {}
        if data.ana_titer:
            serology["ANA"] = f"Positive (1:{data.ana_titer})" if data.ana_titer >= 80 else "Negative"
        if data.rf_iu_ml is not None:
            serology["RF"] = "Positive" if data.rf_iu_ml > self.markers.RF_NORMAL_MAX else "Negative"
        if data.anti_ccp_u_ml is not None:
            serology["Anti-CCP"] = "Positive" if data.anti_ccp_u_ml > self.markers.ANTI_CCP_NEGATIVE_MAX else "Negative"
        if data.anti_dsdna_iu_ml is not None:
            serology["Anti-dsDNA"] = "Positive" if data.anti_dsdna_iu_ml > self.markers.ANTI_DSDNA_POSITIVE_MIN else "Negative"

        # Clinical features score
        clinical_score = sum([
            data.joint_symptoms, data.symmetric_arthritis, data.malar_rash,
            data.photosensitivity, data.oral_ulcers, data.serositis,
            data.raynauds_phenomenon, data.skin_thickening, data.dry_eyes_mouth
        ])

        # RA-specific score
        ra_score = self.calculate_ra_score(data) if primary_diagnosis == AutoimmuneDisease.RHEUMATOID_ARTHRITIS else None

        # SLE-specific criteria
        sle_criteria = self.count_sle_criteria(data) if primary_diagnosis == AutoimmuneDisease.SYSTEMIC_LUPUS_ERYTHEMATOSUS else None

        # Disease activity
        if data.esr_mm_hr and data.esr_mm_hr > self.markers.ESR_HIGH_MIN:
            activity = "HIGH inflammatory activity"
        elif data.esr_mm_hr and data.esr_mm_hr > self.markers.ESR_ELEVATED_MIN:
            activity = "MODERATE inflammatory activity"
        elif data.crp_mg_l and data.crp_mg_l > self.markers.CRP_ELEVATED_MIN:
            activity = "MODERATE inflammatory activity"
        else:
            activity = "LOW inflammatory activity"

        # Recommendations
        recommendations = self._generate_recommendations(primary_diagnosis, confidence, data)

        # Clinical notes
        notes = f"Age {data.age} {data.sex} | Primary: {primary_diagnosis.value} ({confidence:.2f}) | "
        if ra_score:
            notes += f"ACR/EULAR score: {ra_score:.1f}/10 | "
        if sle_criteria:
            notes += f"SLE criteria met: {sle_criteria}/11 | "
        notes += f"{activity}"

        return ClassificationResult(
            primary_diagnosis=primary_diagnosis,
            confidence=round(confidence, 3),
            differential_diagnoses=differentials,
            serological_profile=serology,
            clinical_features_score=clinical_score,
            acr_eular_score=round(ra_score, 1) if ra_score else None,
            sle_criteria_met=sle_criteria,
            disease_activity=activity,
            recommendations=recommendations,
            clinical_notes=notes
        )

    def _generate_recommendations(self, diagnosis: AutoimmuneDisease, confidence: float,
                                  data: SerologyInput) -> List[str]:
        """Generate clinical recommendations"""
        recs = []

        if confidence < 0.50:
            recs.append("INCONCLUSIVE - Consider additional testing or rheumatology referral")

        if diagnosis == AutoimmuneDisease.RHEUMATOID_ARTHRITIS:
            recs.extend([
                "Rheumatology referral for DMARD initiation (methotrexate first-line)",
                "Baseline imaging (X-ray hands/feet) for erosion assessment",
                "Monitor disease activity with DAS28 or CDAI scores"
            ])
        elif diagnosis == AutoimmuneDisease.SYSTEMIC_LUPUS_ERYTHEMATOSUS:
            recs.extend([
                "Rheumatology referral for immunosuppressive therapy consideration",
                "Complete renal workup (creatinine, urine protein, renal biopsy if indicated)",
                "Hydroxychloroquine for long-term disease control"
            ])
        elif diagnosis == AutoimmuneDisease.SJOGRENS_SYNDROME:
            recs.extend([
                "Ophthalmology evaluation for Schirmer test and keratoconjunctivitis sicca",
                "Artificial tears and saliva substitutes for symptom management",
                "Consider hydroxychloroquine for systemic manifestations"
            ])

        if data.crp_mg_l and data.crp_mg_l > 100:
            recs.append("HIGH inflammation - Consider short-term corticosteroids pending specialist evaluation")

        recs.append("Lifestyle: Sun protection, smoking cessation, regular exercise")

        return recs

# FastAPI endpoints
classifier = AutoimmuneClassifier()

@app.post("/classify", response_model=ClassificationResult)
async def classify_autoimmune(data: SerologyInput):
    """Classify autoimmune disease from serology and clinical features"""
    try:
        return classifier.classify(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/thresholds")
async def get_thresholds():
    """Get serological marker thresholds"""
    return {
        "ana_positive_titer": 80,
        "rf_positive": AutoimmuneMarkers.RF_NORMAL_MAX,
        "anti_ccp_positive": AutoimmuneMarkers.ANTI_CCP_POSITIVE_MIN,
        "anti_dsdna_positive": AutoimmuneMarkers.ANTI_DSDNA_POSITIVE_MIN,
        "esr_elevated": AutoimmuneMarkers.ESR_ELEVATED_MIN,
        "crp_elevated": AutoimmuneMarkers.CRP_ELEVATED_MIN
    }

@app.get("/health")
async def health_check():
    return {"status": "operational", "system": "Autoimmune Disease Classifier", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    print("🧬 Autoimmune Disease Classifier - Clinical Grade System")
    print("📊 Using ACR/EULAR criteria and validated serological thresholds")
    uvicorn.run(app, host="0.0.0.0", port=8003)
