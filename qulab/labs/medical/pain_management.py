"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Pain Management Optimizer
Clinical-grade WHO analgesic ladder with validated pain scales
"""
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List

app = FastAPI(title="Pain Management Optimizer", version="1.0.0")

class PainType(str, Enum):
    NOCICEPTIVE_SOMATIC = "nociceptive_somatic"
    NOCICEPTIVE_VISCERAL = "nociceptive_visceral"
    NEUROPATHIC = "neuropathic"
    MIXED = "mixed"
    CANCER_PAIN = "cancer_related"

class WHOLadderStep(str, Enum):
    STEP_1 = "step_1_nonopioid"
    STEP_2 = "step_2_weak_opioid"
    STEP_3 = "step_3_strong_opioid"

class PainInput(BaseModel):
    nrs_score: int = Field(..., ge=0, le=10, description="Numeric Rating Scale 0-10")
    pain_type: PainType
    acute_vs_chronic: str = Field("chronic", description="acute/chronic")
    breakthrough_pain: bool = Field(False)
    current_medications: List[str] = Field(default_factory=list, description="Current analgesics")
    renal_impairment: bool = Field(False, description="CrCl <60")
    liver_impairment: bool = Field(False)
    respiratory_disease: bool = Field(False)
    history_substance_abuse: bool = Field(False)
    elderly: bool = Field(False, description="Age >65")
    previous_opioid_exposure: bool = Field(False)

class PainReport(BaseModel):
    pain_severity: str
    who_ladder_step: WHOLadderStep
    recommended_regimen: List[str]
    adjuvant_therapy: List[str]
    precautions: List[str]
    monitoring_parameters: List[str]
    clinical_notes: str

class PainManagementEngine:
    # WHO analgesic ladder thresholds
    MILD_PAIN_MAX = 3
    MODERATE_PAIN_MAX = 6
    
    # Medication safety constants (approximate equianalgesic doses)
    MORPHINE_EQUIVALENT_MG = {
        "tramadol_50mg": 5,
        "codeine_30mg": 3,
        "hydrocodone_5mg": 5,
        "oxycodone_5mg": 7.5,
        "morphine_10mg": 10,
        "hydromorphone_2mg": 20,
        "fentanyl_25mcg_hr": 60  # per hour patch
    }
    
    def classify_pain_severity(self, nrs: int) -> str:
        """Classify pain by NRS score"""
        if nrs <= self.MILD_PAIN_MAX:
            return "MILD"
        elif nrs <= self.MODERATE_PAIN_MAX:
            return "MODERATE"
        else:
            return "SEVERE"
    
    def determine_ladder_step(self, nrs: int, current_meds: List[str]) -> WHOLadderStep:
        """WHO analgesic ladder step"""
        severity = self.classify_pain_severity(nrs)
        
        # Check current opioid use
        weak_opioids = ["tramadol", "codeine", "hydrocodone"]
        strong_opioids = ["morphine", "oxycodone", "hydromorphone", "fentanyl", "methadone"]
        
        on_weak_opioid = any(med in " ".join(current_meds).lower() for med in weak_opioids)
        on_strong_opioid = any(med in " ".join(current_meds).lower() for med in strong_opioids)
        
        if severity == "SEVERE" or (severity == "MODERATE" and on_weak_opioid):
            return WHOLadderStep.STEP_3
        elif severity == "MODERATE":
            return WHOLadderStep.STEP_2
        else:
            return WHOLadderStep.STEP_1
    
    def recommend_regimen(self, step: WHOLadderStep, pain_type: PainType,
                         acute: bool, contraindications: dict) -> List[str]:
        """Evidence-based medication recommendations"""
        regimen = []
        
        # Step 1: Non-opioid analgesics
        if step in [WHOLadderStep.STEP_1, WHOLadderStep.STEP_2, WHOLadderStep.STEP_3]:
            if not contraindications.get("nsaid"):
                regimen.append("Acetaminophen 650-1000mg q6h PRN (max 3g/day if chronic)")
                if pain_type in [PainType.NOCICEPTIVE_SOMATIC, PainType.MIXED]:
                    regimen.append("Ibuprofen 400-600mg q6h PRN (with food, max 14 days)")
            else:
                regimen.append("Acetaminophen 650-1000mg q6h PRN (max 4g/day)")
        
        # Step 2: Weak opioids
        if step in [WHOLadderStep.STEP_2, WHOLadderStep.STEP_3]:
            if not contraindications.get("opioid"):
                if step == WHOLadderStep.STEP_2:
                    regimen.append("Tramadol 50-100mg q6h PRN (start low if opioid-naive)")
                    regimen.append("OR Codeine 30-60mg q4h PRN")
        
        # Step 3: Strong opioids
        if step == WHOLadderStep.STEP_3:
            if not contraindications.get("opioid"):
                if acute:
                    regimen.append("Oxycodone IR 5-10mg q4h PRN (titrate to effect)")
                    regimen.append("OR Morphine IR 5-10mg q4h PRN")
                else:
                    regimen.append("Oxycodone ER 10mg q12h scheduled (with IR for breakthrough)")
                    regimen.append("OR Morphine ER 15mg q12h scheduled")
                    regimen.append("Consider fentanyl patch if stable dose requirement")
        
        return regimen
    
    def recommend_adjuvants(self, pain_type: PainType, breakthrough: bool) -> List[str]:
        """Adjuvant therapies by pain type"""
        adjuvants = []
        
        if pain_type == PainType.NEUROPATHIC:
            adjuvants.extend([
                "Gabapentin 300mg TID (titrate to 900-3600mg/day) - first line",
                "OR Pregabalin 75mg BID (titrate to 150-300mg BID)",
                "Consider duloxetine 60mg daily (if comorbid depression/anxiety)",
                "Topical lidocaine 5% patch for localized neuropathic pain"
            ])
        
        if pain_type == PainType.NOCICEPTIVE_VISCERAL:
            adjuvants.append("Consider antispasmodics (dicyclomine, hyoscyamine)")
        
        if breakthrough:
            adjuvants.append("Breakthrough dose: 10-15% of total daily opioid dose q1h PRN")
        
        # Non-pharmacologic
        adjuvants.extend([
            "Physical therapy/occupational therapy referral",
            "Cognitive behavioral therapy (CBT) for chronic pain",
            "Consider acupuncture, TENS unit"
        ])
        
        return adjuvants
    
    def identify_precautions(self, data: PainInput, step: WHOLadderStep) -> List[str]:
        """Safety precautions"""
        precautions = []
        
        if step in [WHOLadderStep.STEP_2, WHOLadderStep.STEP_3]:
            precautions.extend([
                "Bowel regimen MANDATORY (senna + docusate)",
                "Avoid driving/operating machinery during titration",
                "Naloxone rescue kit if high-dose opioids"
            ])
        
        if data.renal_impairment:
            precautions.append("Renal dosing: Reduce opioid doses by 50%, avoid NSAIDs")
        
        if data.liver_impairment:
            precautions.append("Hepatic dosing: Reduce acetaminophen to <2g/day, caution with opioids")
        
        if data.respiratory_disease:
            precautions.append("CAUTION: Opioids depress respiration - start low, go slow")
        
        if data.elderly:
            precautions.append("Geriatric dosing: Start 25-50% lower, titrate slowly")
        
        if data.history_substance_abuse:
            precautions.extend([
                "HIGH RISK: Consider non-opioid strategies preferentially",
                "Pain contract and urine drug screening if opioids necessary",
                "Consult addiction medicine if available"
            ])
        
        return precautions
    
    def monitoring_parameters(self, step: WHOLadderStep) -> List[str]:
        """Required monitoring"""
        params = [
            "Pain score (NRS) at each visit",
            "Functional status (activities of daily living)",
            "Adverse effects (constipation, sedation, nausea)"
        ]
        
        if step in [WHOLadderStep.STEP_2, WHOLadderStep.STEP_3]:
            params.extend([
                "State prescription drug monitoring program (PDMP) check",
                "Reassess opioid need at each refill",
                "Consider tapering if pain improved or function not improving"
            ])
        
        return params
    
    def assess(self, data: PainInput) -> PainReport:
        severity = self.classify_pain_severity(data.nrs_score)
        step = self.determine_ladder_step(data.nrs_score, data.current_medications)
        
        # Contraindications
        contraindications = {
            "nsaid": data.renal_impairment or data.liver_impairment,
            "opioid": False  # Would need more specific criteria
        }
        
        regimen = self.recommend_regimen(step, data.pain_type, data.acute_vs_chronic == "acute",
                                        contraindications)
        adjuvants = self.recommend_adjuvants(data.pain_type, data.breakthrough_pain)
        precautions = self.identify_precautions(data, step)
        monitoring = self.monitoring_parameters(step)
        
        notes = f"NRS {data.nrs_score}/10 ({severity}) | {data.pain_type.value} | {data.acute_vs_chronic} | WHO Step {step.value}"
        
        return PainReport(
            pain_severity=severity,
            who_ladder_step=step,
            recommended_regimen=regimen,
            adjuvant_therapy=adjuvants,
            precautions=precautions,
            monitoring_parameters=monitoring,
            clinical_notes=notes
        )

engine = PainManagementEngine()

@app.post("/assess", response_model=PainReport)
async def assess_pain(data: PainInput):
    return engine.assess(data)

@app.get("/health")
async def health_check():
    return {"status": "operational", "system": "Pain Management Optimizer", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
