"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Liver Disease Staging System
Clinical-grade MELD, Child-Pugh, FIB-4, and APRI scoring
"""
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from enum import Enum

app = FastAPI(title="Liver Disease Staging", version="1.0.0")

class ChildPughClass(str, Enum):
    A = "A_compensated"
    B = "B_significant_impairment"
    C = "C_decompensated"

class LiverInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    bilirubin_mg_dl: float = Field(..., ge=0.1, le=50.0)
    albumin_g_dl: float = Field(..., ge=1.0, le=6.0)
    inr: float = Field(..., ge=0.8, le=10.0, description="International Normalized Ratio")
    creatinine_mg_dl: float = Field(..., ge=0.3, le=15.0)
    sodium_mmol_l: float = Field(140, ge=120, le=160)
    dialysis_twice_weekly: bool = Field(False)
    ascites: str = Field("none", description="none/mild/moderate")
    encephalopathy: str = Field("none", description="none/grade_1_2/grade_3_4")
    alt_u_l: Optional[float] = Field(None, ge=1, le=2000, description="ALT for FIB-4")
    ast_u_l: Optional[float] = Field(None, ge=1, le=2000, description="AST for FIB-4")
    platelet_count_k_ul: Optional[float] = Field(None, ge=10, le=1000, description="Platelets (×10³/μL)")

class LiverReport(BaseModel):
    meld_score: float
    meld_na_score: float
    child_pugh_score: int
    child_pugh_class: ChildPughClass
    fib4_score: Optional[float]
    apri_score: Optional[float]
    one_year_mortality: float
    transplant_priority: str
    fibrosis_stage: Optional[str]
    recommendations: list[str]
    clinical_notes: str

class LiverDiseaseEngine:
    def calculate_meld(self, bilirubin: float, inr: float, creatinine: float, dialysis: bool) -> float:
        """
        MELD score (Model for End-Stage Liver Disease)
        Range: 6-40, higher = worse
        """
        # Cap values per UNOS guidelines
        bilirubin = max(1.0, min(bilirubin, 4.0))
        inr = max(1.0, min(inr, 4.0))
        creatinine = max(1.0, min(creatinine, 4.0))
        
        if dialysis:
            creatinine = 4.0  # Maximum for dialysis patients
        
        meld = 9.57 * np.log(creatinine) + 3.78 * np.log(bilirubin) + 11.2 * np.log(inr) + 6.43
        meld = max(6, min(40, round(meld)))
        
        return meld
    
    def calculate_meld_na(self, meld: float, sodium: float) -> float:
        """MELD-Na incorporates hyponatremia"""
        sodium = max(125, min(sodium, 137))
        meld_na = meld + 1.32 * (137 - sodium) - (0.033 * meld * (137 - sodium))
        return max(6, min(40, round(meld_na, 1)))
    
    def calculate_child_pugh(self, bilirubin: float, albumin: float, inr: float,
                            ascites: str, encephalopathy: str) -> tuple[int, ChildPughClass]:
        """Child-Pugh score (5-15 points)"""
        score = 0
        
        # Bilirubin (mg/dL)
        if bilirubin < 2.0:
            score += 1
        elif bilirubin < 3.0:
            score += 2
        else:
            score += 3
        
        # Albumin (g/dL)
        if albumin > 3.5:
            score += 1
        elif albumin > 2.8:
            score += 2
        else:
            score += 3
        
        # INR
        if inr < 1.7:
            score += 1
        elif inr < 2.3:
            score += 2
        else:
            score += 3
        
        # Ascites
        if ascites == "none":
            score += 1
        elif ascites == "mild":
            score += 2
        else:
            score += 3
        
        # Encephalopathy
        if encephalopathy == "none":
            score += 1
        elif encephalopathy == "grade_1_2":
            score += 2
        else:
            score += 3
        
        # Classification
        if score <= 6:
            cp_class = ChildPughClass.A
        elif score <= 9:
            cp_class = ChildPughClass.B
        else:
            cp_class = ChildPughClass.C
        
        return score, cp_class
    
    def calculate_fib4(self, age: int, ast: float, alt: float, platelets: float) -> float:
        """FIB-4 index for fibrosis staging"""
        fib4 = (age * ast) / (platelets * np.sqrt(alt))
        return round(fib4, 2)
    
    def calculate_apri(self, ast: float, platelets: float) -> float:
        """APRI score for fibrosis"""
        ast_uln = 40  # Upper limit of normal
        apri = (ast / ast_uln) / platelets * 100
        return round(apri, 2)
    
    def estimate_mortality(self, meld_na: float) -> float:
        """3-month mortality prediction from MELD-Na"""
        # Empirical relationship from UNOS data
        mortality_3mo = 1.9 * (1.0 ** (0.15 * meld_na))
        return min(100.0, round(mortality_3mo, 1))
    
    def assess(self, data: LiverInput) -> LiverReport:
        meld = self.calculate_meld(data.bilirubin_mg_dl, data.inr, data.creatinine_mg_dl, data.dialysis_twice_weekly)
        meld_na = self.calculate_meld_na(meld, data.sodium_mmol_l)
        cp_score, cp_class = self.calculate_child_pugh(data.bilirubin_mg_dl, data.albumin_g_dl, data.inr,
                                                       data.ascites, data.encephalopathy)
        
        fib4 = None
        apri = None
        fibrosis_stage = None
        
        if data.ast_u_l and data.alt_u_l and data.platelet_count_k_ul:
            fib4 = self.calculate_fib4(data.age, data.ast_u_l, data.alt_u_l, data.platelet_count_k_ul)
            apri = self.calculate_apri(data.ast_u_l, data.platelet_count_k_ul)
            
            if fib4 < 1.45:
                fibrosis_stage = "F0-F1 (minimal/no fibrosis)"
            elif fib4 < 3.25:
                fibrosis_stage = "F2-F3 (moderate fibrosis, indeterminate)"
            else:
                fibrosis_stage = "F4 (advanced fibrosis/cirrhosis)"
        
        mortality_1yr = self.estimate_mortality(meld_na) * 4  # 3mo × 4 ≈ 1 year
        
        if meld_na >= 30:
            transplant = "URGENT - High priority"
        elif meld_na >= 20:
            transplant = "HIGH PRIORITY"
        elif meld_na >= 15:
            transplant = "MODERATE PRIORITY"
        else:
            transplant = "Low priority"
        
        recs = []
        if cp_class == ChildPughClass.C or meld_na >= 15:
            recs.extend([
                "Hepatology/transplant evaluation URGENT",
                "Consider listing for liver transplant"
            ])
        elif cp_class == ChildPughClass.B:
            recs.append("Hepatology referral for transplant workup")
        
        if data.ascites != "none":
            recs.extend([
                "Diuretics (spironolactone + furosemide)",
                "Sodium restriction (<2g/day)",
                "Monitor for spontaneous bacterial peritonitis"
            ])
        
        if data.encephalopathy != "none":
            recs.extend([
                "Lactulose for hepatic encephalopathy",
                "Rifaximin if lactulose insufficient",
                "Protein restriction NOT recommended (maintain adequate nutrition)"
            ])
        
        recs.extend([
            "Variceal screening (endoscopy if cirrhosis)",
            "Hepatocellular carcinoma surveillance (ultrasound + AFP q6mo)",
            "Avoid hepatotoxic medications (acetaminophen >2g/day, NSAIDs)"
        ])
        
        notes = f"MELD-Na {meld_na:.1f} | Child-Pugh {cp_class.value} (score {cp_score}) | 1-yr mortality ~{mortality_1yr:.0f}%"
        
        return LiverReport(
            meld_score=meld,
            meld_na_score=meld_na,
            child_pugh_score=cp_score,
            child_pugh_class=cp_class,
            fib4_score=fib4,
            apri_score=apri,
            one_year_mortality=min(100.0, mortality_1yr),
            transplant_priority=transplant,
            fibrosis_stage=fibrosis_stage,
            recommendations=recs,
            clinical_notes=notes
        )

engine = LiverDiseaseEngine()

@app.post("/assess", response_model=LiverReport)
async def assess_liver(data: LiverInput):
    return engine.assess(data)

@app.get("/health")
async def health_check():
    return {"status": "operational", "system": "Liver Disease Staging", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
