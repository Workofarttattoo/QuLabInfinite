"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Lung Function Analyzer
Clinical-grade spirometry interpretation with ATS/ERS guidelines
"""
import numpy as np
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
from enum import Enum

app = FastAPI(title="Lung Function Analyzer", version="1.0.0")

class Pattern(str, Enum):
    NORMAL = "normal"
    OBSTRUCTIVE = "obstructive"
    RESTRICTIVE = "restrictive"
    MIXED = "mixed"

class Severity(str, Enum):
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    MODERATELY_SEVERE = "moderately_severe"
    SEVERE = "severe"
    VERY_SEVERE = "very_severe"

class SpirometryInput(BaseModel):
    age: int = Field(..., ge=5, le=100)
    sex: str = Field(..., description="M/F")
    height_cm: float = Field(..., ge=100, le=220)
    race: str = Field("caucasian", description="caucasian/african_american/asian/hispanic")
    fev1_liters: float = Field(..., ge=0.1, le=8.0, description="FEV1 measured")
    fvc_liters: float = Field(..., ge=0.1, le=10.0, description="FVC measured")
    dlco_ml_min_mmhg: Optional[float] = Field(None, ge=1, le=50, description="DLCO if available")
    hemoglobin_g_dl: Optional[float] = Field(None, ge=5, le=20)

class LungReport(BaseModel):
    fev1_percent_predicted: float
    fvc_percent_predicted: float
    fev1_fvc_ratio: float
    pattern: Pattern
    severity: Severity
    dlco_percent_predicted: Optional[float]
    interpretation: str
    recommendations: list[str]
    clinical_notes: str

class LungFunctionEngine:
    def predict_fev1(self, age: int, sex: str, height_cm: float, race: str) -> float:
        """GLI-2012 predicted FEV1 (simplified)"""
        height_m = height_cm / 100
        if sex == "M":
            pred = (0.5536 * height_m ** 2.3466) * np.exp(-0.02426 * age)
        else:
            pred = (0.4333 * height_m ** 2.1814) * np.exp(-0.01974 * age)
        
        # Race adjustment factors
        if race == "african_american":
            pred *= 0.88
        elif race == "asian":
            pred *= 0.94
        
        return pred
    
    def predict_fvc(self, age: int, sex: str, height_cm: float, race: str) -> float:
        """GLI-2012 predicted FVC"""
        height_m = height_cm / 100
        if sex == "M":
            pred = (0.6552 * height_m ** 2.5080) * np.exp(-0.01916 * age)
        else:
            pred = (0.5196 * height_m ** 2.3670) * np.exp(-0.01587 * age)
        
        if race == "african_american":
            pred *= 0.88
        elif race == "asian":
            pred *= 0.94
        
        return pred
    
    def predict_dlco(self, age: int, sex: str, height_cm: float) -> float:
        """Predicted DLCO (simplified Miller equation)"""
        if sex == "M":
            pred = 0.416 * height_cm - 0.219 * age - 26.3
        else:
            pred = 0.256 * height_cm - 0.144 * age - 8.36
        return pred
    
    def classify_severity_fev1(self, fev1_pct: float) -> Severity:
        """ATS/ERS severity based on FEV1 % predicted"""
        if fev1_pct >= 80:
            return Severity.NORMAL
        elif fev1_pct >= 70:
            return Severity.MILD
        elif fev1_pct >= 60:
            return Severity.MODERATE
        elif fev1_pct >= 50:
            return Severity.MODERATELY_SEVERE
        elif fev1_pct >= 35:
            return Severity.SEVERE
        else:
            return Severity.VERY_SEVERE
    
    def assess(self, data: SpirometryInput) -> LungReport:
        pred_fev1 = self.predict_fev1(data.age, data.sex, data.height_cm, data.race)
        pred_fvc = self.predict_fvc(data.age, data.sex, data.height_cm, data.race)
        
        fev1_pct = (data.fev1_liters / pred_fev1) * 100
        fvc_pct = (data.fvc_liters / pred_fvc) * 100
        ratio = data.fev1_liters / data.fvc_liters
        
        # Pattern classification per ATS/ERS
        if ratio < 0.70 and fev1_pct < 80:
            if fvc_pct < 80:
                pattern = Pattern.MIXED
            else:
                pattern = Pattern.OBSTRUCTIVE
        elif fvc_pct < 80 and ratio >= 0.70:
            pattern = Pattern.RESTRICTIVE
        else:
            pattern = Pattern.NORMAL
        
        severity = self.classify_severity_fev1(fev1_pct)
        
        dlco_pct = None
        if data.dlco_ml_min_mmhg:
            pred_dlco = self.predict_dlco(data.age, data.sex, data.height_cm)
            dlco_adjusted = data.dlco_ml_min_mmhg
            if data.hemoglobin_g_dl:
                # Hemoglobin correction
                dlco_adjusted = data.dlco_ml_min_mmhg * (10.22 + data.hemoglobin_g_dl) / (1.7 * data.hemoglobin_g_dl)
            dlco_pct = (dlco_adjusted / pred_dlco) * 100
        
        # Interpretation
        interp_parts = []
        if pattern == Pattern.OBSTRUCTIVE:
            interp_parts.append(f"{severity.value.upper()} obstructive pattern (COPD/asthma likely)")
        elif pattern == Pattern.RESTRICTIVE:
            interp_parts.append(f"Restrictive pattern - consider ILD, neuromuscular disease, obesity")
        elif pattern == Pattern.MIXED:
            interp_parts.append(f"{severity.value.upper()} mixed obstruction + restriction")
        else:
            interp_parts.append("Normal spirometry")
        
        if dlco_pct:
            if dlco_pct < 60:
                interp_parts.append(f"DLCO severely reduced ({dlco_pct:.0f}%) - parenchymal/vascular disease")
            elif dlco_pct < 80:
                interp_parts.append(f"DLCO mildly reduced ({dlco_pct:.0f}%)")
        
        interpretation = " | ".join(interp_parts)
        
        recs = []
        if pattern == Pattern.OBSTRUCTIVE:
            recs.extend([
                "Bronchodilator trial (albuterol, assess reversibility)",
                "Consider inhaled corticosteroids if asthma features",
                "Smoking cessation counseling if applicable",
                "Pulmonary rehabilitation if COPD"
            ])
        elif pattern == Pattern.RESTRICTIVE:
            recs.extend([
                "High-resolution CT chest to evaluate for ILD",
                "Consider neuromuscular evaluation if weakness present",
                "Pulmonology referral"
            ])
        
        if dlco_pct and dlco_pct < 60:
            recs.append("6-minute walk test with oximetry - assess for hypoxemia")
        
        recs.append("Repeat spirometry annually or if symptoms worsen")
        
        notes = f"FEV1 {fev1_pct:.0f}% pred | FVC {fvc_pct:.0f}% pred | Ratio {ratio:.2f} | {pattern.value}"
        
        return LungReport(
            fev1_percent_predicted=round(fev1_pct, 1),
            fvc_percent_predicted=round(fvc_pct, 1),
            fev1_fvc_ratio=round(ratio, 3),
            pattern=pattern,
            severity=severity,
            dlco_percent_predicted=round(dlco_pct, 1) if dlco_pct else None,
            interpretation=interpretation,
            recommendations=recs,
            clinical_notes=notes
        )

engine = LungFunctionEngine()

@app.post("/assess", response_model=LungReport)
async def assess_lung(data: SpirometryInput):
    return engine.assess(data)

@app.get("/health")
async def health_check():
    return {"status": "operational", "system": "Lung Function Analyzer", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
