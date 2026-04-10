"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Wound Healing Optimizer
Clinical-grade TIME framework (Tissue/Infection/Moisture/Edge) assessment
"""
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

app = FastAPI(title="Wound Healing Optimizer", version="1.0.0")

class WoundType(str, Enum):
    PRESSURE_ULCER = "pressure_ulcer"
    VENOUS_ULCER = "venous_ulcer"
    ARTERIAL_ULCER = "arterial_ulcer"
    DIABETIC_ULCER = "diabetic_ulcer"
    SURGICAL = "surgical_wound"
    TRAUMATIC = "traumatic_wound"
    BURN = "burn_wound"

class HealingStage(str, Enum):
    ACUTE_INFLAMMATORY = "acute_inflammatory"
    PROLIFERATIVE = "proliferative"
    REMODELING = "remodeling"
    CHRONIC_STALLED = "chronic_stalled"

class WoundInput(BaseModel):
    wound_type: WoundType
    length_cm: float = Field(..., ge=0.1, le=100.0)
    width_cm: float = Field(..., ge=0.1, le=100.0)
    depth_cm: float = Field(0.0, ge=0.0, le=20.0)
    tissue_necrotic_percent: int = Field(0, ge=0, le=100)
    tissue_slough_percent: int = Field(0, ge=0, le=100)
    tissue_granulation_percent: int = Field(0, ge=0, le=100)
    tissue_epithelialization_percent: int = Field(0, ge=0, le=100)
    infection_clinical_signs: bool = Field(False)
    exudate_amount: str = Field("moderate", description="none/minimal/moderate/heavy")
    wound_edges_attached: bool = Field(True)
    periwound_intact: bool = Field(True)
    patient_age: int = Field(..., ge=0, le=120)
    diabetes: bool = Field(False)
    vascular_disease: bool = Field(False)
    immunosuppressed: bool = Field(False)
    smoking: bool = Field(False)
    hba1c_percent: Optional[float] = Field(None, ge=4.0, le=15.0)
    albumin_g_dl: Optional[float] = Field(None, ge=1.0, le=6.0)

class WoundReport(BaseModel):
    healing_stage: HealingStage
    time_score: int
    healing_trajectory_weeks: float
    tissue_quality: str
    infection_risk: str
    moisture_balance: str
    edge_advancement: str
    interventions: List[str]
    healing_impediments: List[str]
    clinical_notes: str

class WoundHealingEngine:
    def __init__(self):
        self.TIME_MAX_SCORE = 16  # 4 domains × 4 points max
    
    def calculate_time_score(self, data: WoundInput) -> tuple[int, Dict]:
        """TIME framework scoring"""
        scores = {}
        # T: Tissue (0-4)
        if data.tissue_granulation_percent >= 75:
            scores['tissue'] = 4
        elif data.tissue_necrotic_percent > 50:
            scores['tissue'] = 0
        elif data.tissue_slough_percent > 50:
            scores['tissue'] = 1
        else:
            scores['tissue'] = 2
        
        # I: Infection (0-4)
        if data.infection_clinical_signs:
            scores['infection'] = 0
        else:
            scores['infection'] = 4
        
        # M: Moisture (0-4)
        if data.exudate_amount == "none":
            scores['moisture'] = 2
        elif data.exudate_amount == "minimal":
            scores['moisture'] = 4
        elif data.exudate_amount == "moderate":
            scores['moisture'] = 3
        else:
            scores['moisture'] = 1
        
        # E: Edge (0-4)
        if data.tissue_epithelialization_percent >= 75:
            scores['edge'] = 4
        elif not data.wound_edges_attached:
            scores['edge'] = 0
        else:
            scores['edge'] = 2
        
        return sum(scores.values()), scores
    
    def predict_healing_time(self, data: WoundInput, time_score: int) -> float:
        """Estimate healing time in weeks"""
        area = data.length_cm * data.width_cm
        volume = area * data.depth_cm if data.depth_cm > 0 else area
        
        base_weeks = np.sqrt(volume) * 0.5
        score_multiplier = (self.TIME_MAX_SCORE - time_score + 1) / self.TIME_MAX_SCORE
        
        if data.diabetes:
            base_weeks *= 1.5
        if data.vascular_disease:
            base_weeks *= 1.8
        if data.smoking:
            base_weeks *= 1.3
        if data.immunosuppressed:
            base_weeks *= 1.4
        if data.patient_age > 65:
            base_weeks *= 1.2
        
        return base_weeks * score_multiplier
    
    def assess(self, data: WoundInput) -> WoundReport:
        time_score, scores = self.calculate_time_score(data)
        healing_weeks = self.predict_healing_time(data, time_score)
        
        if time_score >= 14:
            stage = HealingStage.PROLIFERATIVE
        elif time_score >= 10:
            stage = HealingStage.ACUTE_INFLAMMATORY
        elif healing_weeks > 12:
            stage = HealingStage.CHRONIC_STALLED
        else:
            stage = HealingStage.REMODELING
        
        interventions = []
        impediments = []
        
        if scores['tissue'] < 3:
            interventions.append("Debridement needed (enzymatic or sharp)")
        if scores['infection'] == 0:
            interventions.append("Antimicrobial therapy - culture and sensitivity")
            impediments.append("Active infection")
        if scores['moisture'] < 3:
            if data.exudate_amount == "heavy":
                interventions.append("Absorptive dressing (foam, alginate)")
            else:
                interventions.append("Moisture-retentive dressing (hydrogel)")
        if scores['edge'] < 2:
            interventions.append("Edge advancement therapy - consider growth factors")
        
        if data.diabetes and (not data.hba1c_percent or data.hba1c_percent > 7.0):
            impediments.append("Uncontrolled diabetes")
        if data.albumin_g_dl and data.albumin_g_dl < 3.0:
            impediments.append("Malnutrition (low albumin)")
        
        notes = f"{data.wound_type.value} | {data.length_cm}×{data.width_cm}×{data.depth_cm}cm | TIME: {time_score}/16"
        
        return WoundReport(
            healing_stage=stage,
            time_score=time_score,
            healing_trajectory_weeks=round(healing_weeks, 1),
            tissue_quality=f"Granulation {data.tissue_granulation_percent}%, Necrotic {data.tissue_necrotic_percent}%",
            infection_risk="HIGH" if scores['infection'] == 0 else "LOW",
            moisture_balance=f"{data.exudate_amount.upper()} exudate",
            edge_advancement=f"Epithelialization {data.tissue_epithelialization_percent}%",
            interventions=interventions,
            healing_impediments=impediments,
            clinical_notes=notes
        )

engine = WoundHealingEngine()

@app.post("/assess", response_model=WoundReport)
async def assess_wound(data: WoundInput):
    return engine.assess(data)

@app.get("/health")
async def health_check():
    return {"status": "operational", "system": "Wound Healing Optimizer", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
