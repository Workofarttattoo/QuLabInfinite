"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Sepsis Early Warning System
Clinical-grade qSOFA, SOFA, and NEWS2 scoring with validated thresholds
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

app = FastAPI(title="Sepsis Early Warning System", version="1.0.0")

# Clinical constants from Sepsis-3 definitions and validated early warning scores
class SepsisConstants:
    """Validated sepsis criteria and early warning thresholds"""
    # qSOFA (quick Sequential Organ Failure Assessment) - 2+ indicates sepsis risk
    QSOFA_THRESHOLD = 2  # Out of 3 criteria
    
    # SOFA (Sequential Organ Failure Assessment) score
    # Sepsis defined as infection + SOFA ≥2
    SOFA_SEPSIS_THRESHOLD = 2
    SOFA_SEPTIC_SHOCK_THRESHOLD = 6
    
    # Vital sign thresholds for qSOFA
    RESPIRATORY_RATE_HIGH = 22  # breaths/min
    SYSTOLIC_BP_LOW = 100  # mmHg
    GCS_LOW = 15  # Glasgow Coma Scale (normal=15, altered if <15)
    
    # Lactate thresholds (mmol/L)
    LACTATE_NORMAL_MAX = 2.0
    LACTATE_ELEVATED = 2.1
    LACTATE_SEVERE = 4.0
    
    # NEWS2 (National Early Warning Score 2) - UK standard
    NEWS2_LOW_RISK_MAX = 4
    NEWS2_MEDIUM_RISK_MAX = 6
    NEWS2_HIGH_RISK_MIN = 7
    
    # White blood cell count (cells/μL)
    WBC_NORMAL_MIN = 4000
    WBC_NORMAL_MAX = 11000
    WBC_LEUKOCYTOSIS = 12000
    WBC_LEUKOPENIA = 4000
    
    # Temperature thresholds (°C)
    TEMP_HYPOTHERMIA = 36.0
    TEMP_NORMAL_MIN = 36.1
    TEMP_NORMAL_MAX = 38.0
    TEMP_FEVER = 38.1
    
    # Heart rate (bpm)
    HR_BRADYCARDIA = 40
    HR_NORMAL_MIN = 51
    HR_NORMAL_MAX = 90
    HR_TACHYCARDIA = 111

class SepsisRisk(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class VitalsInput(BaseModel):
    """Patient vital signs and laboratory values"""
    respiratory_rate: int = Field(..., description="Breaths per minute", ge=4, le=60)
    systolic_bp: int = Field(..., description="Systolic blood pressure (mmHg)", ge=40, le=250)
    heart_rate: int = Field(..., description="Heart rate (bpm)", ge=30, le=200)
    temperature_celsius: float = Field(..., description="Body temperature (°C)", ge=32.0, le=42.0)
    spo2_percent: int = Field(..., description="Oxygen saturation (%)", ge=50, le=100)
    supplemental_o2: bool = Field(False, description="On supplemental oxygen")
    gcs_score: int = Field(15, description="Glasgow Coma Scale (3-15)", ge=3, le=15)
    lactate_mmol_l: Optional[float] = Field(None, description="Serum lactate (mmol/L)", ge=0.1, le=20.0)
    wbc_count: Optional[int] = Field(None, description="White blood cell count (cells/μL)", ge=0, le=100000)
    creatinine_mg_dl: Optional[float] = Field(None, description="Creatinine (mg/dL)", ge=0.1, le=15.0)
    bilirubin_mg_dl: Optional[float] = Field(None, description="Bilirubin (mg/dL)", ge=0.0, le=30.0)
    platelets_k_ul: Optional[int] = Field(None, description="Platelet count (×10³/μL)", ge=0, le=1000)
    suspected_infection: bool = Field(False, description="Clinical suspicion of infection")
    altered_mental_status: bool = Field(False, description="Altered mental status noted")

class SepsisWarningReport(BaseModel):
    """Sepsis early warning assessment"""
    sepsis_risk: SepsisRisk
    qsofa_score: int
    sofa_score: int
    news2_score: int
    lactate_status: str
    hemodynamic_status: str
    organ_dysfunction: List[str]
    recommendations: List[str]
    time_to_intervention_minutes: int
    clinical_notes: str

class SepsisEarlyWarning:
    """Clinical-grade sepsis screening using validated scores"""
    
    def __init__(self):
        self.constants = SepsisConstants()
    
    def calculate_qsofa(self, rr: int, sbp: int, gcs: int) -> int:
        """Quick SOFA score (0-3)"""
        score = 0
        if rr >= self.constants.RESPIRATORY_RATE_HIGH:
            score += 1
        if sbp <= self.constants.SYSTOLIC_BP_LOW:
            score += 1
        if gcs < self.constants.GCS_LOW:
            score += 1
        return score
    
    def calculate_sofa(self, data: VitalsInput) -> tuple[int, List[str]]:
        """Sequential Organ Failure Assessment (0-24)"""
        score = 0
        dysfunctions = []
        
        # Respiration (PaO2/FiO2 ratio approximation from SpO2)
        if data.spo2_percent < 92:
            score += 2
            dysfunctions.append("Respiratory")
        elif data.spo2_percent < 96 and data.supplemental_o2:
            score += 1
            dysfunctions.append("Respiratory")
        
        # Coagulation (platelets)
        if data.platelets_k_ul:
            if data.platelets_k_ul < 50:
                score += 2
                dysfunctions.append("Coagulation")
            elif data.platelets_k_ul < 100:
                score += 1
                dysfunctions.append("Coagulation")
        
        # Liver (bilirubin)
        if data.bilirubin_mg_dl:
            if data.bilirubin_mg_dl >= 12.0:
                score += 4
                dysfunctions.append("Hepatic")
            elif data.bilirubin_mg_dl >= 6.0:
                score += 3
                dysfunctions.append("Hepatic")
            elif data.bilirubin_mg_dl >= 2.0:
                score += 2
                dysfunctions.append("Hepatic")
        
        # Cardiovascular (hypotension)
        if data.systolic_bp < 70:
            score += 4
            dysfunctions.append("Cardiovascular")
        elif data.systolic_bp <= 100:
            score += 2
            dysfunctions.append("Cardiovascular")
        
        # CNS (GCS)
        if data.gcs_score <= 6:
            score += 4
            dysfunctions.append("Neurologic")
        elif data.gcs_score <= 9:
            score += 3
            dysfunctions.append("Neurologic")
        elif data.gcs_score <= 12:
            score += 2
            dysfunctions.append("Neurologic")
        elif data.gcs_score <= 14:
            score += 1
            dysfunctions.append("Neurologic")
        
        # Renal (creatinine)
        if data.creatinine_mg_dl:
            if data.creatinine_mg_dl >= 5.0:
                score += 4
                dysfunctions.append("Renal")
            elif data.creatinine_mg_dl >= 3.5:
                score += 3
                dysfunctions.append("Renal")
            elif data.creatinine_mg_dl >= 2.0:
                score += 2
                dysfunctions.append("Renal")
            elif data.creatinine_mg_dl >= 1.2:
                score += 1
                dysfunctions.append("Renal")
        
        return score, list(set(dysfunctions))
    
    def calculate_news2(self, data: VitalsInput) -> int:
        """National Early Warning Score 2 (0-20)"""
        score = 0
        
        # Respiratory rate
        if data.respiratory_rate <= 8:
            score += 3
        elif data.respiratory_rate <= 11:
            score += 1
        elif data.respiratory_rate >= 25:
            score += 3
        elif data.respiratory_rate >= 21:
            score += 2
        
        # SpO2 (Scale 1 if on O2, Scale 2 if not)
        if not data.supplemental_o2:
            if data.spo2_percent <= 91:
                score += 3
            elif data.spo2_percent <= 93:
                score += 2
            elif data.spo2_percent <= 95:
                score += 1
        else:
            if data.spo2_percent <= 83:
                score += 3
            elif data.spo2_percent <= 85:
                score += 2
            elif data.spo2_percent <= 87:
                score += 1
            score += 2  # On supplemental O2
        
        # Systolic BP
        if data.systolic_bp <= 90:
            score += 3
        elif data.systolic_bp <= 100:
            score += 2
        elif data.systolic_bp <= 110:
            score += 1
        elif data.systolic_bp >= 220:
            score += 3
        
        # Heart rate
        if data.heart_rate <= 40:
            score += 3
        elif data.heart_rate <= 50:
            score += 1
        elif data.heart_rate >= 131:
            score += 3
        elif data.heart_rate >= 111:
            score += 2
        elif data.heart_rate >= 91:
            score += 1
        
        # Consciousness (AVPU or GCS)
        if data.altered_mental_status or data.gcs_score < 15:
            score += 3
        
        # Temperature
        if data.temperature_celsius <= 35.0:
            score += 3
        elif data.temperature_celsius <= 36.0:
            score += 1
        elif data.temperature_celsius >= 39.1:
            score += 2
        elif data.temperature_celsius >= 38.1:
            score += 1
        
        return score
    
    def assess_sepsis_risk(self, data: VitalsInput) -> SepsisWarningReport:
        """Comprehensive sepsis early warning assessment"""
        
        # Calculate scores
        qsofa = self.calculate_qsofa(data.respiratory_rate, data.systolic_bp, data.gcs_score)
        sofa, organ_dysfunctions = self.calculate_sofa(data)
        news2 = self.calculate_news2(data)
        
        # Lactate assessment
        if data.lactate_mmol_l:
            if data.lactate_mmol_l >= self.constants.LACTATE_SEVERE:
                lactate_status = "SEVERE hyperlactatemia"
            elif data.lactate_mmol_l > self.constants.LACTATE_NORMAL_MAX:
                lactate_status = "ELEVATED lactate"
            else:
                lactate_status = "Normal lactate"
        else:
            lactate_status = "Not measured"
        
        # Hemodynamic status
        if data.systolic_bp < 70:
            hemo_status = "SHOCK - severe hypotension"
        elif data.systolic_bp <= 90:
            hemo_status = "HYPOTENSION - pressors likely needed"
        elif data.systolic_bp <= 100:
            hemo_status = "Borderline BP - monitor closely"
        else:
            hemo_status = "Hemodynamically stable"
        
        # Determine overall risk
        if (qsofa >= self.constants.QSOFA_THRESHOLD and data.suspected_infection) or \
           sofa >= self.constants.SOFA_SEPTIC_SHOCK_THRESHOLD or \
           (data.lactate_mmol_l and data.lactate_mmol_l >= self.constants.LACTATE_SEVERE):
            risk = SepsisRisk.CRITICAL
            time_to_intervention = 0  # Immediate
        elif (qsofa >= self.constants.QSOFA_THRESHOLD) or \
             (sofa >= self.constants.SOFA_SEPSIS_THRESHOLD and data.suspected_infection) or \
             news2 >= self.constants.NEWS2_HIGH_RISK_MIN:
            risk = SepsisRisk.HIGH
            time_to_intervention = 15
        elif news2 >= self.constants.NEWS2_MEDIUM_RISK_MAX or qsofa >= 1:
            risk = SepsisRisk.MODERATE
            time_to_intervention = 60
        else:
            risk = SepsisRisk.LOW
            time_to_intervention = 240
        
        # Recommendations
        recommendations = self._generate_recommendations(risk, qsofa, sofa, data)
        
        # Clinical notes
        notes = f"qSOFA: {qsofa}/3 | SOFA: {sofa}/24 | NEWS2: {news2}/20 | "
        notes += f"BP {data.systolic_bp}/{data.heart_rate} | SpO2 {data.spo2_percent}% | "
        notes += f"RR {data.respiratory_rate} | Temp {data.temperature_celsius}°C"
        if data.lactate_mmol_l:
            notes += f" | Lactate {data.lactate_mmol_l} mmol/L"
        
        return SepsisWarningReport(
            sepsis_risk=risk,
            qsofa_score=qsofa,
            sofa_score=sofa,
            news2_score=news2,
            lactate_status=lactate_status,
            hemodynamic_status=hemo_status,
            organ_dysfunction=organ_dysfunctions,
            recommendations=recommendations,
            time_to_intervention_minutes=time_to_intervention,
            clinical_notes=notes
        )
    
    def _generate_recommendations(self, risk: SepsisRisk, qsofa: int, sofa: int,
                                  data: VitalsInput) -> List[str]:
        """Generate clinical recommendations"""
        recs = []
        
        if risk == SepsisRisk.CRITICAL:
            recs.extend([
                "CODE SEPSIS - Activate sepsis protocol IMMEDIATELY",
                "Start IV antibiotics within 1 hour (broad-spectrum)",
                "30 mL/kg IV fluid bolus for resuscitation",
                "Serial lactate measurements (target <2 mmol/L)",
                "Vasopressors if MAP <65 mmHg after fluid resuscitation",
                "ICU admission"
            ])
        elif risk == SepsisRisk.HIGH:
            recs.extend([
                "Activate sepsis protocol - bundle compliance critical",
                "Blood cultures x2 before antibiotics",
                "IV antibiotics within 1 hour",
                "IV fluid resuscitation (30 mL/kg if hypotensive)",
                "Measure lactate if not done",
                "Reassess within 15 minutes"
            ])
        elif risk == SepsisRisk.MODERATE:
            recs.extend([
                "Repeat vital signs every 30 minutes",
                "Consider blood cultures and labs",
                "Establish IV access",
                "Source control evaluation",
                "Escalate if deterioration"
            ])
        else:
            recs.append("Continue standard monitoring")
        
        # Organ-specific recommendations
        if data.spo2_percent < 92:
            recs.append("Supplemental oxygen to maintain SpO2 ≥94%")
        
        if data.lactate_mmol_l and data.lactate_mmol_l > 4.0:
            recs.append("Aggressive fluid resuscitation for lactate clearance")
        
        return recs

# FastAPI endpoints
engine = SepsisEarlyWarning()

@app.post("/assess", response_model=SepsisWarningReport)
async def assess_sepsis(data: VitalsInput):
    """Sepsis early warning assessment"""
    try:
        return engine.assess_sepsis_risk(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/thresholds")
async def get_thresholds():
    return {
        "qsofa_threshold": SepsisConstants.QSOFA_THRESHOLD,
        "sofa_sepsis": SepsisConstants.SOFA_SEPSIS_THRESHOLD,
        "lactate_elevated": SepsisConstants.LACTATE_ELEVATED,
        "news2_high_risk": SepsisConstants.NEWS2_HIGH_RISK_MIN
    }

@app.get("/health")
async def health_check():
    return {"status": "operational", "system": "Sepsis Early Warning System", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    print("🚨 Sepsis Early Warning System - Clinical Grade")
    print("📊 qSOFA, SOFA, NEWS2 scoring with Sepsis-3 criteria")
    uvicorn.run(app, host="0.0.0.0", port=8004)
