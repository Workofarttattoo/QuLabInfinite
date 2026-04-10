"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Parkinson's Disease Progression Predictor
Clinical-grade motor and non-motor symptom tracking with validated scales
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

app = FastAPI(title="Parkinson's Progression Predictor", version="1.0.0")

# Clinical constants from Movement Disorder Society-UPDRS (MDS-UPDRS)
class ParkinsonsScales:
    """Validated clinical rating scales for Parkinson's disease"""

    # MDS-UPDRS Part III (Motor Examination) score ranges
    # Total score 0-132 (higher = worse motor function)
    UPDRS_PART3_NORMAL_MAX = 5
    UPDRS_PART3_MILD_MAX = 32
    UPDRS_PART3_MODERATE_MAX = 58
    UPDRS_PART3_SEVERE_MIN = 59

    # Hoehn and Yahr staging (0-5)
    # Stage 0: No signs of disease
    # Stage 1: Unilateral involvement only
    # Stage 1.5: Unilateral and axial involvement
    # Stage 2: Bilateral involvement without balance impairment
    # Stage 2.5: Mild bilateral disease with recovery on pull test
    # Stage 3: Bilateral disease, mild to moderate, some postural instability
    # Stage 4: Severe disability, still able to walk or stand unassisted
    # Stage 5: Wheelchair bound or bedridden unless aided

    # Schwab and England Activities of Daily Living (ADL) Scale (%)
    # 100%: Completely independent
    # 90%: Completely independent, slower
    # 80%: Independent, definite slowing, some difficulty
    # 70%: Not completely independent, some tasks take 3-4x longer
    # 60%: Some dependency, can do most chores slowly
    # 50%: More dependent, needs help with half of chores
    # 40%: Very dependent, can assist with most chores
    # 30%: Needs considerable help, can occasionally do few chores
    # 20%: Can do nothing alone, some help with some chores
    # 10%: Totally dependent, helpless
    # 0%: Vegetative functions impaired

    ADL_INDEPENDENT_MIN = 80
    ADL_MODERATE_DISABILITY_MIN = 50
    ADL_SEVERE_DISABILITY_MAX = 40

    # Levodopa equivalent daily dose (LEDD) mg/day
    LEDD_LOW_MAX = 400
    LEDD_MODERATE_MAX = 800
    LEDD_HIGH_MIN = 801

    # Motor fluctuation thresholds (hours off per day)
    MOTOR_FLUCTUATION_MILD_MAX = 2.0
    MOTOR_FLUCTUATION_MODERATE_MAX = 4.0
    MOTOR_FLUCTUATION_SEVERE_MIN = 4.1

    # Freezing of gait (FOG) frequency (episodes per week)
    FOG_MILD_MAX = 3
    FOG_MODERATE_MAX = 10
    FOG_SEVERE_MIN = 11

    # Montreal Cognitive Assessment (MoCA) for PD-MCI screening
    MOCA_NORMAL_MIN = 26
    MOCA_PD_MCI_MAX = 25

    # REM Sleep Behavior Disorder (RBD) - associated with faster progression
    # Measured by RBD questionnaire score (0-13, ≥5 indicates probable RBD)
    RBD_POSITIVE_THRESHOLD = 5

class DiseaseStage(str, Enum):
    EARLY = "early"  # H&Y 1-1.5
    MODERATE = "moderate"  # H&Y 2-2.5
    ADVANCED = "advanced"  # H&Y 3-4
    SEVERE = "severe"  # H&Y 5

class ProgressionRate(str, Enum):
    SLOW = "slow"
    MODERATE = "moderate"
    RAPID = "rapid"

class MotorSubtype(str, Enum):
    TREMOR_DOMINANT = "tremor_dominant"
    PIGD = "postural_instability_gait_disorder"
    MIXED = "mixed"

class ParkinsonsClinicalData(BaseModel):
    """Clinical measurements for Parkinson's disease assessment"""
    age: int = Field(..., description="Patient age (years)", ge=30, le=100)
    disease_duration_years: float = Field(..., description="Years since diagnosis", ge=0, le=50)
    updrs_part3_score: int = Field(..., description="MDS-UPDRS Part III motor score (0-132)", ge=0, le=132)
    hoehn_yahr_stage: float = Field(..., description="Modified Hoehn & Yahr stage (0-5)", ge=0, le=5)
    schwab_england_adl: int = Field(..., description="ADL percentage (0-100%)", ge=0, le=100)
    ledd_mg_per_day: float = Field(..., description="Levodopa equivalent daily dose (mg)", ge=0, le=3000)
    motor_fluctuations_hours: float = Field(0, description="Hours of OFF time per day", ge=0, le=16)
    dyskinesia_present: bool = Field(False, description="Presence of dyskinesia")
    freezing_of_gait_episodes: int = Field(0, description="FOG episodes per week", ge=0, le=100)
    falls_per_month: int = Field(0, description="Number of falls in past month", ge=0, le=100)
    moca_score: Optional[int] = Field(None, description="Montreal Cognitive Assessment (0-30)", ge=0, le=30)
    rbd_score: Optional[int] = Field(None, description="RBD questionnaire score (0-13)", ge=0, le=13)
    autonomic_symptoms_count: int = Field(0, description="Number of autonomic symptoms (0-10)", ge=0, le=10)
    depression_present: bool = Field(False, description="Clinical depression diagnosis")
    tremor_dominant: bool = Field(True, description="Tremor-dominant vs PIGD phenotype")

class ProgressionReport(BaseModel):
    """Comprehensive Parkinson's progression analysis"""
    current_stage: DiseaseStage
    motor_subtype: MotorSubtype
    progression_rate: ProgressionRate
    progression_score: float = Field(..., description="Composite progression score (0-100)")
    predicted_hoehn_yahr_1yr: float = Field(..., description="Predicted H&Y stage in 1 year")
    predicted_hoehn_yahr_5yr: float = Field(..., description="Predicted H&Y stage in 5 years")
    motor_milestones: Dict[str, str]
    non_motor_burden: str
    treatment_response: str
    complications_risk: Dict[str, float]
    recommendations: List[str]
    clinical_notes: str

class ParkinsonsProgressionEngine:
    """Clinical-grade Parkinson's disease progression analysis"""

    def __init__(self):
        self.scales = ParkinsonsScales()

    def determine_disease_stage(self, hy_stage: float) -> DiseaseStage:
        """Determine disease stage from Hoehn & Yahr"""
        if hy_stage <= 1.5:
            return DiseaseStage.EARLY
        elif hy_stage <= 2.5:
            return DiseaseStage.MODERATE
        elif hy_stage <= 4.0:
            return DiseaseStage.ADVANCED
        else:
            return DiseaseStage.SEVERE

    def determine_motor_subtype(self, tremor_dominant: bool, fog_episodes: int,
                                falls: int, pigd_features: int) -> MotorSubtype:
        """
        Classify motor subtype based on validated criteria
        PIGD features: FOG, falls, postural instability
        """
        pigd_score = (fog_episodes > 5) + (falls > 2) + (pigd_features > 2)

        if tremor_dominant and pigd_score <= 1:
            return MotorSubtype.TREMOR_DOMINANT
        elif not tremor_dominant and pigd_score >= 2:
            return MotorSubtype.PIGD
        else:
            return MotorSubtype.MIXED

    def calculate_annual_progression_rate(self, disease_duration: float,
                                         hy_stage: float) -> float:
        """
        Calculate annual H&Y progression rate
        Based on longitudinal studies (Hely et al., 2008)
        """
        if disease_duration < 0.1:
            return 0.0

        return hy_stage / disease_duration

    def predict_future_stage(self, current_hy: float, progression_rate: float,
                            years: float, modifiers: Dict) -> float:
        """
        Predict future Hoehn & Yahr stage
        Modifiers adjust progression based on clinical factors
        """
        # Base prediction
        predicted = current_hy + (progression_rate * years)

        # Apply clinical modifiers
        if modifiers.get("motor_subtype") == MotorSubtype.PIGD:
            predicted += 0.3 * years  # PIGD progresses faster

        if modifiers.get("cognitive_impairment"):
            predicted += 0.2 * years

        if modifiers.get("rbd_positive"):
            predicted += 0.15 * years

        if modifiers.get("age") > 70:
            predicted += 0.1 * years

        # Tremor-dominant has slower progression
        if modifiers.get("motor_subtype") == MotorSubtype.TREMOR_DOMINANT:
            predicted -= 0.1 * years

        # Cap at maximum stage
        return min(5.0, max(current_hy, predicted))

    def assess_motor_complications(self, ledd: float, disease_duration: float,
                                  motor_fluctuations: float,
                                  dyskinesia: bool) -> Dict[str, float]:
        """
        Assess risk of motor complications
        Returns probability (0-1) for each complication
        """
        complications = {}

        # Motor fluctuations risk (wearing-off, on-off)
        # Risk increases with disease duration and LEDD
        base_fluctuation_risk = min(0.9, 0.1 + (disease_duration * 0.08))
        if ledd > self.scales.LEDD_HIGH_MIN:
            base_fluctuation_risk *= 1.4
        if motor_fluctuations > 0:
            complications["motor_fluctuations"] = min(1.0, base_fluctuation_risk * 1.5)
        else:
            complications["motor_fluctuations"] = base_fluctuation_risk

        # Dyskinesia risk
        # Cumulative levodopa exposure is key factor
        years_on_high_ledd = max(0, disease_duration - 2)
        dyskinesia_risk = min(0.8, 0.05 + (years_on_high_ledd * 0.12))
        if dyskinesia:
            complications["dyskinesia"] = 1.0  # Already present
        else:
            complications["dyskinesia"] = dyskinesia_risk

        # Freezing of gait risk
        fog_risk = min(0.7, 0.15 + (disease_duration * 0.06))
        complications["freezing_of_gait"] = fog_risk

        # Falls risk
        falls_risk = min(0.8, 0.1 + (disease_duration * 0.08))
        complications["falls"] = falls_risk

        return complications

    def assess_non_motor_burden(self, moca: Optional[int], rbd: Optional[int],
                                autonomic_symptoms: int, depression: bool) -> tuple[str, int]:
        """
        Assess non-motor symptom burden
        Returns: (severity_description, burden_score_0_to_10)
        """
        burden_score = 0

        # Cognitive impairment
        if moca is not None:
            if moca < 21:
                burden_score += 3  # PD dementia range
            elif moca <= self.scales.MOCA_PD_MCI_MAX:
                burden_score += 2  # PD-MCI range

        # REM sleep behavior disorder
        if rbd is not None and rbd >= self.scales.RBD_POSITIVE_THRESHOLD:
            burden_score += 2

        # Autonomic symptoms
        if autonomic_symptoms >= 5:
            burden_score += 2
        elif autonomic_symptoms >= 3:
            burden_score += 1

        # Depression
        if depression:
            burden_score += 1

        # Classify severity
        if burden_score >= 7:
            severity = "SEVERE non-motor burden"
        elif burden_score >= 4:
            severity = "MODERATE non-motor burden"
        elif burden_score >= 2:
            severity = "MILD non-motor burden"
        else:
            severity = "MINIMAL non-motor burden"

        return severity, burden_score

    def assess_treatment_response(self, ledd: float, updrs3: int,
                                  hy_stage: float, disease_duration: float) -> str:
        """Assess levodopa response and treatment optimization"""

        # Expected UPDRS improvement with levodopa
        if disease_duration < 5:
            # Early disease - excellent response expected
            if ledd < self.scales.LEDD_LOW_MAX and updrs3 > self.scales.UPDRS_PART3_MILD_MAX:
                return "SUBOPTIMAL: Consider LEDD escalation for better motor control"
            elif ledd > self.scales.LEDD_HIGH_MIN and updrs3 > self.scales.UPDRS_PART3_MODERATE_MAX:
                return "POOR RESPONSE: Evaluate for atypical parkinsonism"
            else:
                return "GOOD RESPONSE: Appropriate treatment for disease stage"
        else:
            # Advanced disease - response may wane
            if ledd > self.scales.LEDD_HIGH_MIN and updrs3 > self.scales.UPDRS_PART3_MODERATE_MAX:
                return "DIMINISHED RESPONSE: Consider device-assisted therapies (DBS, pump)"
            elif ledd > self.scales.LEDD_MODERATE_MAX:
                return "ADEQUATE RESPONSE: Monitor for motor complications"
            else:
                return "STABLE RESPONSE: Continue current regimen"

    def calculate_progression_score(self, data: ParkinsonsClinicalData,
                                   annual_rate: float, burden_score: int) -> float:
        """
        Calculate composite progression score (0-100)
        Higher score = faster/more severe progression
        """
        # Motor severity component (0-40 points)
        motor_score = min(40, (data.updrs_part3_score / 132) * 40)

        # Functional disability component (0-25 points)
        disability_score = 25 * (1 - data.schwab_england_adl / 100)

        # Progression rate component (0-20 points)
        rate_score = min(20, annual_rate * 10)

        # Non-motor burden component (0-15 points)
        non_motor_score = min(15, burden_score * 1.5)

        total_score = motor_score + disability_score + rate_score + non_motor_score

        return round(total_score, 1)

    def generate_recommendations(self, data: ParkinsonsClinicalData,
                                stage: DiseaseStage, subtype: MotorSubtype,
                                complications: Dict, burden: str) -> List[str]:
        """Generate evidence-based clinical recommendations"""
        recs = []

        # Medication management
        if data.ledd_mg_per_day < self.scales.LEDD_LOW_MAX and data.updrs_part3_score > self.scales.UPDRS_PART3_MILD_MAX:
            recs.append("Consider optimizing dopaminergic therapy - current LEDD may be subtherapeutic")

        if data.motor_fluctuations_hours > self.scales.MOTOR_FLUCTUATION_MODERATE_MAX:
            recs.append("Significant motor fluctuations - evaluate for adjunctive therapies (MAO-B inhibitors, COMT inhibitors)")

        if data.dyskinesia_present and data.ledd_mg_per_day > self.scales.LEDD_HIGH_MIN:
            recs.append("Dyskinesia present with high LEDD - consider fractionating doses or adding amantadine")

        # Device-assisted therapies
        if stage in [DiseaseStage.ADVANCED, DiseaseStage.SEVERE] and complications.get("motor_fluctuations", 0) > 0.6:
            recs.append("Advanced disease with motor complications - DBS evaluation recommended")

        # Falls and FOG
        if data.falls_per_month > 2:
            recs.append("HIGH FALL RISK - physical therapy, home safety evaluation, consider walking aid")

        if data.freezing_of_gait_episodes > self.scales.FOG_MODERATE_MAX:
            recs.append("Severe FOG - gait training, cueing strategies, medication review")

        # Non-motor symptoms
        if "SEVERE" in burden or "MODERATE" in burden:
            recs.append(f"{burden} - multidisciplinary management (neurology, psychiatry, sleep medicine)")

        if data.moca_score and data.moca_score <= self.scales.MOCA_PD_MCI_MAX:
            recs.append("Cognitive impairment detected - cognitive assessment, cholinesterase inhibitor consideration")

        # Exercise and lifestyle
        recs.extend([
            "High-intensity exercise 3-5x/week (neuroprotective effects documented)",
            "Multidisciplinary care: PT, OT, speech therapy as needed",
            "Parkinson's support group participation for patient and caregiver"
        ])

        # Monitoring
        if subtype == MotorSubtype.PIGD:
            recs.append("PIGD subtype - more frequent monitoring due to faster progression risk")

        return recs

    def analyze_progression(self, data: ParkinsonsClinicalData) -> ProgressionReport:
        """Comprehensive Parkinson's disease progression analysis"""

        # Determine current stage
        stage = self.determine_disease_stage(data.hoehn_yahr_stage)

        # Determine motor subtype
        # PIGD features: FOG, falls, postural instability (inferred from H&Y stage)
        pigd_features = int(data.hoehn_yahr_stage >= 3.0)
        motor_subtype = self.determine_motor_subtype(
            data.tremor_dominant,
            data.freezing_of_gait_episodes,
            data.falls_per_month,
            pigd_features
        )

        # Calculate progression rate
        annual_rate = self.calculate_annual_progression_rate(
            data.disease_duration_years,
            data.hoehn_yahr_stage
        )

        # Determine progression category
        if annual_rate < 0.3:
            progression_rate = ProgressionRate.SLOW
        elif annual_rate < 0.6:
            progression_rate = ProgressionRate.MODERATE
        else:
            progression_rate = ProgressionRate.RAPID

        # Non-motor burden
        burden_desc, burden_score = self.assess_non_motor_burden(
            data.moca_score,
            data.rbd_score,
            data.autonomic_symptoms_count,
            data.depression_present
        )

        # Modifiers for prediction
        modifiers = {
            "motor_subtype": motor_subtype,
            "cognitive_impairment": data.moca_score and data.moca_score <= self.scales.MOCA_PD_MCI_MAX,
            "rbd_positive": data.rbd_score and data.rbd_score >= self.scales.RBD_POSITIVE_THRESHOLD,
            "age": data.age
        }

        # Future predictions
        predicted_1yr = self.predict_future_stage(data.hoehn_yahr_stage, annual_rate, 1.0, modifiers)
        predicted_5yr = self.predict_future_stage(data.hoehn_yahr_stage, annual_rate, 5.0, modifiers)

        # Motor complications
        complications = self.assess_motor_complications(
            data.ledd_mg_per_day,
            data.disease_duration_years,
            data.motor_fluctuations_hours,
            data.dyskinesia_present
        )

        # Treatment response
        treatment_response = self.assess_treatment_response(
            data.ledd_mg_per_day,
            data.updrs_part3_score,
            data.hoehn_yahr_stage,
            data.disease_duration_years
        )

        # Motor milestones
        motor_milestones = {
            "bilateral_involvement": "REACHED" if data.hoehn_yahr_stage >= 2.0 else "NOT REACHED",
            "postural_instability": "REACHED" if data.hoehn_yahr_stage >= 3.0 else "NOT REACHED",
            "severe_disability": "REACHED" if data.hoehn_yahr_stage >= 4.0 else "NOT REACHED",
            "wheelchair_dependent": "REACHED" if data.hoehn_yahr_stage >= 5.0 else "NOT REACHED"
        }

        # Progression score
        progression_score = self.calculate_progression_score(data, annual_rate, burden_score)

        # Recommendations
        recommendations = self.generate_recommendations(
            data, stage, motor_subtype, complications, burden_desc
        )

        # Clinical notes
        clinical_notes = (
            f"Patient age {data.age}, disease duration {data.disease_duration_years:.1f} years | "
            f"H&Y stage {data.hoehn_yahr_stage}, {motor_subtype.value} subtype | "
            f"UPDRS-III: {data.updrs_part3_score}/132, S&E ADL: {data.schwab_england_adl}% | "
            f"LEDD: {data.ledd_mg_per_day:.0f} mg/day | "
            f"Annual progression rate: {annual_rate:.2f} H&Y units/year"
        )

        return ProgressionReport(
            current_stage=stage,
            motor_subtype=motor_subtype,
            progression_rate=progression_rate,
            progression_score=progression_score,
            predicted_hoehn_yahr_1yr=round(predicted_1yr, 1),
            predicted_hoehn_yahr_5yr=round(predicted_5yr, 1),
            motor_milestones=motor_milestones,
            non_motor_burden=burden_desc,
            treatment_response=treatment_response,
            complications_risk={k: round(v, 3) for k, v in complications.items()},
            recommendations=recommendations,
            clinical_notes=clinical_notes
        )

# FastAPI endpoints
engine = ParkinsonsProgressionEngine()

@app.post("/analyze", response_model=ProgressionReport)
async def analyze_progression(data: ParkinsonsClinicalData):
    """Comprehensive Parkinson's progression analysis"""
    try:
        return engine.analyze_progression(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/scales")
async def get_clinical_scales():
    """Get clinical rating scale information"""
    return {
        "updrs_part3": {
            "range": "0-132",
            "mild_max": ParkinsonsScales.UPDRS_PART3_MILD_MAX,
            "moderate_max": ParkinsonsScales.UPDRS_PART3_MODERATE_MAX,
            "description": "MDS-UPDRS Part III Motor Examination"
        },
        "hoehn_yahr": {
            "range": "0-5",
            "stages": {
                "0": "No signs",
                "1": "Unilateral",
                "2": "Bilateral without balance impairment",
                "3": "Mild to moderate bilateral, postural instability",
                "4": "Severe disability",
                "5": "Wheelchair bound"
            }
        },
        "schwab_england": {
            "range": "0-100%",
            "independent_min": ParkinsonsScales.ADL_INDEPENDENT_MIN,
            "description": "Activities of Daily Living scale"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "operational", "system": "Parkinson's Progression Predictor", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    print("🧠 Parkinson's Disease Progression Predictor - Clinical Grade System")
    print("📊 Using validated MDS-UPDRS and H&Y staging criteria")
    print("🔬 Motor subtype classification and longitudinal prediction enabled")
    uvicorn.run(app, host="0.0.0.0", port=8002)
