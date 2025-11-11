"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Kidney Function Calculator
Clinical-grade eGFR using CKD-EPI and MDRD equations, CKD staging per KDIGO
"""
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum

app = FastAPI(title="Kidney Function Calculator", version="1.0.0")

class CKDStage(str, Enum):
    G1 = "G1_normal_high"
    G2 = "G2_mildly_decreased"
    G3A = "G3a_mild_moderate"
    G3B = "G3b_moderate_severe"
    G4 = "G4_severely_decreased"
    G5 = "G5_kidney_failure"

class KidneyInput(BaseModel):
    age: int = Field(..., ge=18, le=120)
    sex: str = Field(..., description="M/F")
    race: str = Field("other", description="black/other")
    creatinine_mg_dl: float = Field(..., ge=0.3, le=15.0)
    cystatin_c_mg_l: Optional[float] = Field(None, ge=0.3, le=8.0)
    albumin_mg_g_cr: Optional[float] = Field(None, ge=0, le=5000, description="Urine albumin-creatinine ratio")

class KidneyReport(BaseModel):
    egfr_ckd_epi: float
    egfr_mdrd: float
    ckd_stage: CKDStage
    albuminuria_stage: str
    risk_category: str
    progression_risk: str
    recommendations: list[str]
    clinical_notes: str

class KidneyFunctionEngine:
    # KDIGO CKD staging (eGFR mL/min/1.73m²)
    STAGE_THRESHOLDS = {
        CKDStage.G5: 15,
        CKDStage.G4: 30,
        CKDStage.G3B: 45,
        CKDStage.G3A: 60,
        CKDStage.G2: 90
    }
    
    def calculate_ckd_epi_2021(self, creat_mg_dl: float, age: int, sex: str, race: str) -> float:
        """
        CKD-EPI 2021 equation (race-free, most current)
        Reference: Inker LA et al. NEJM 2021
        """
        # Convert creatinine to μmol/L then back to standardized form
        kappa = 0.7 if sex == "F" else 0.9
        alpha = -0.241 if sex == "F" else -0.302
        
        min_ratio = min(creat_mg_dl / kappa, 1.0)
        max_ratio = max(creat_mg_dl / kappa, 1.0)
        
        egfr = 142 * (min_ratio ** alpha) * (max_ratio ** -1.200) * (0.9938 ** age)
        
        if sex == "F":
            egfr *= 1.012
        
        return egfr
    
    def calculate_mdrd(self, creat_mg_dl: float, age: int, sex: str, race: str) -> float:
        """
        MDRD equation (legacy, for comparison)
        """
        egfr = 175 * (creat_mg_dl ** -1.154) * (age ** -0.203)
        
        if sex == "F":
            egfr *= 0.742
        if race == "black":
            egfr *= 1.212
        
        return egfr
    
    def stage_ckd(self, egfr: float) -> CKDStage:
        """KDIGO staging"""
        if egfr < self.STAGE_THRESHOLDS[CKDStage.G5]:
            return CKDStage.G5
        elif egfr < self.STAGE_THRESHOLDS[CKDStage.G4]:
            return CKDStage.G4
        elif egfr < self.STAGE_THRESHOLDS[CKDStage.G3B]:
            return CKDStage.G3B
        elif egfr < self.STAGE_THRESHOLDS[CKDStage.G3A]:
            return CKDStage.G3A
        elif egfr < self.STAGE_THRESHOLDS[CKDStage.G2]:
            return CKDStage.G2
        else:
            return CKDStage.G1
    
    def stage_albuminuria(self, acr: Optional[float]) -> tuple[str, str]:
        """KDIGO albuminuria staging"""
        if acr is None:
            return "Unknown", "A?"
        elif acr < 30:
            return "Normal to mildly increased", "A1"
        elif acr < 300:
            return "Moderately increased", "A2"
        else:
            return "Severely increased", "A3"
    
    def assess_risk(self, stage: CKDStage, alb_stage: str) -> str:
        """KDIGO risk stratification (combined GFR + albuminuria)"""
        risk_matrix = {
            (CKDStage.G1, "A1"): "Low risk",
            (CKDStage.G1, "A2"): "Moderately increased risk",
            (CKDStage.G1, "A3"): "High risk",
            (CKDStage.G2, "A1"): "Low risk",
            (CKDStage.G2, "A2"): "Moderately increased risk",
            (CKDStage.G2, "A3"): "High risk",
            (CKDStage.G3A, "A1"): "Moderately increased risk",
            (CKDStage.G3A, "A2"): "High risk",
            (CKDStage.G3A, "A3"): "Very high risk",
            (CKDStage.G3B, "A1"): "High risk",
            (CKDStage.G3B, "A2"): "Very high risk",
            (CKDStage.G3B, "A3"): "Very high risk",
            (CKDStage.G4, "A1"): "Very high risk",
            (CKDStage.G4, "A2"): "Very high risk",
            (CKDStage.G4, "A3"): "Very high risk",
            (CKDStage.G5, "A1"): "Very high risk",
            (CKDStage.G5, "A2"): "Very high risk",
            (CKDStage.G5, "A3"): "Very high risk"
        }
        return risk_matrix.get((stage, alb_stage), "Unknown risk")
    
    def assess(self, data: KidneyInput) -> KidneyReport:
        egfr_ckd_epi = self.calculate_ckd_epi_2021(data.creatinine_mg_dl, data.age, data.sex, data.race)
        egfr_mdrd = self.calculate_mdrd(data.creatinine_mg_dl, data.age, data.sex, data.race)
        
        stage = self.stage_ckd(egfr_ckd_epi)
        alb_desc, alb_stage = self.stage_albuminuria(data.albumin_mg_g_cr)
        risk = self.assess_risk(stage, alb_stage)
        
        recs = []
        if stage in [CKDStage.G4, CKDStage.G5]:
            recs.extend([
                "Nephrology referral URGENT",
                "Evaluate for renal replacement therapy (dialysis/transplant)",
                "Medication dosing adjustments required"
            ])
        elif stage in [CKDStage.G3A, CKDStage.G3B]:
            recs.extend([
                "Nephrology referral recommended",
                "Review nephrotoxic medications",
                "Blood pressure goal <130/80 mmHg"
            ])
        
        if alb_stage == "A3":
            recs.append("ACE inhibitor or ARB for albuminuria reduction")
        
        recs.extend([
            "Diabetic control if applicable (HbA1c <7%)",
            "Dietary protein restriction (0.8 g/kg/day)",
            "Monitor electrolytes, phosphate, PTH",
            "Repeat eGFR annually or more frequently if declining"
        ])
        
        if egfr_ckd_epi < 45:
            progression = "RAPID progression risk - monitor closely"
        elif egfr_ckd_epi < 60:
            progression = "Moderate progression risk"
        else:
            progression = "Slow progression expected with good control"
        
        notes = f"Age {data.age} {data.sex} | eGFR {egfr_ckd_epi:.1f} mL/min/1.73m² | Stage {stage.value} | {alb_desc}"
        
        return KidneyReport(
            egfr_ckd_epi=round(egfr_ckd_epi, 1),
            egfr_mdrd=round(egfr_mdrd, 1),
            ckd_stage=stage,
            albuminuria_stage=alb_stage,
            risk_category=risk,
            progression_risk=progression,
            recommendations=recs,
            clinical_notes=notes
        )

engine = KidneyFunctionEngine()

@app.post("/assess", response_model=KidneyReport)
async def assess_kidney(data: KidneyInput):
    return engine.assess(data)

@app.get("/health")
async def health_check():
    return {"status": "operational", "system": "Kidney Function Calculator", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)
