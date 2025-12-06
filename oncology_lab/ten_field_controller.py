"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

ten_field_controller - Part of Oncology Lab
"""

"""
Controller for the ten-field tumour microenvironment abstraction.

This component keeps track of simplified proxies for pH, oxygenation, nutrients,
and other factors often discussed in oncology literature. It supports scripted
interventions so experiments can explore hypothetical protocols. The model is
heuristic and should be treated as an exploratory tool rather than a clinical
truth.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class InterventionType(Enum):
    """Types of interventions that can be applied"""
    DIETARY = "dietary"  # Ketogenic diet, fasting, etc.
    PHARMACOLOGICAL = "pharmacological"  # Drugs
    PHYSICAL = "physical"  # Hyperthermia, HBOT, etc.
    COMBINATION = "combination"


@dataclass
class FieldIntervention:
    """
    Single intervention targeting specific field(s)
    """
    name: str
    intervention_type: InterventionType
    target_fields: List[str]  # Which of the 10 fields this affects
    effects: Dict[str, float]  # Field name -> delta value
    onset_hours: float = 1.0  # How quickly it takes effect
    duration_hours: float = 24.0  # How long effects last
    intensity: float = 1.0  # 0-1 scale for strength


@dataclass
class FieldInterventionProtocol:
    """
    Complete intervention protocol with multiple interventions
    Matches ECH0's 3-stage protocol from analysis
    """
    name: str
    description: str
    interventions: List[FieldIntervention] = field(default_factory=list)
    schedule: Dict[float, List[str]] = field(default_factory=dict)  # time_hours -> intervention names

    def add_intervention(self, intervention: FieldIntervention, start_time: float = 0.0):
        """Add an intervention to the protocol"""
        self.interventions.append(intervention)
        if start_time not in self.schedule:
            self.schedule[start_time] = []
        self.schedule[start_time].append(intervention.name)

    def get_active_interventions(self, current_time: float) -> List[FieldIntervention]:
        """Get all interventions active at current time"""
        active = []
        for intervention in self.interventions:
            # Find when this intervention started
            start_times = [t for t, names in self.schedule.items()
                          if intervention.name in names]
            if start_times:
                start_time = max([t for t in start_times if t <= current_time], default=None)
                if start_time is not None:
                    time_since_start = current_time - start_time
                    if time_since_start < intervention.duration_hours:
                        active.append(intervention)
        return active


@dataclass
class InterventionResponse:
    """
    Measured response to intervention
    """
    time_hours: float
    field_values: Dict[str, float]  # Current value of each field
    tumor_cell_count: int
    tumor_viability: float
    metabolic_stress: float
    intervention_efficacy: float  # 0-1 scale


class TenFieldController:
    """
    Controls and monitors the 10 biological fields during interventions
    """

    def __init__(self):
        # Current field values (normal/baseline)
        self.fields = {
            'ph_level': 7.4,
            'oxygen_percent': 21.0,
            'glucose_mm': 5.5,
            'lactate_mm': 1.0,
            'temperature_c': 37.0,
            'ros_um': 0.1,
            'glutamine_mm': 0.6,
            'calcium_um': 100.0,
            'atp_adp_ratio': 10.0,
            'cytokine_pg_ml': 5.0,
        }

        # Baseline values (for reference)
        self.baseline_fields = self.fields.copy()

        # Cancer-promoting values
        self.cancer_fields = {
            'ph_level': 6.7,
            'oxygen_percent': 1.0,
            'glucose_mm': 15.0,
            'lactate_mm': 10.0,
            'temperature_c': 37.0,
            'ros_um': 5.0,
            'glutamine_mm': 2.0,
            'calcium_um': 500.0,
            'atp_adp_ratio': 5.0,
            'cytokine_pg_ml': 50.0,
        }

        # Target remission values (from ECH0 analysis)
        self.remission_fields = {
            'ph_level': 7.4,
            'oxygen_percent': 21.0,
            'glucose_mm': 0.5,  # ECH0 suggested more extreme
            'lactate_mm': 0.5,
            'temperature_c': 39.0,  # Mild hyperthermia baseline
            'ros_um': 2.0,
            'glutamine_mm': 0.2,
            'calcium_um': 150.0,
            'atp_adp_ratio': 12.0,
            'cytokine_pg_ml': 2.0,
        }

        # Response history
        self.history: List[InterventionResponse] = []
        self.time = 0.0

    def apply_cancer_profile(self, overrides: Dict[str, float]):
        """
        Override the default cancer microenvironment with tumour-specific values.
        Used to reflect different tumour phenotypes before interventions begin.
        """
        if not overrides:
            return

        for field_name, value in overrides.items():
            if field_name in self.cancer_fields:
                self.cancer_fields[field_name] = value

    def set_cancer_microenvironment(self):
        """Set fields to cancer-promoting values"""
        self.fields = self.cancer_fields.copy()

    def set_normal_environment(self):
        """Reset to normal physiological values"""
        self.fields = self.baseline_fields.copy()

    def apply_intervention_effects(self, interventions: List[FieldIntervention], dt: float):
        """
        Apply effects of active interventions to fields
        """
        for intervention in interventions:
            # Calculate current effectiveness based on onset
            effectiveness = min(1.0, intervention.intensity)

            # Apply effects to each target field
            for field_name, delta in intervention.effects.items():
                if field_name in self.fields:
                    current = self.fields[field_name]
                    target_change = delta * effectiveness * dt
                    self.fields[field_name] += target_change

            # Clamp fields to realistic ranges
            self.fields['ph_level'] = np.clip(self.fields['ph_level'], 6.0, 8.0)
            self.fields['oxygen_percent'] = np.clip(self.fields['oxygen_percent'], 0.0, 100.0)
            self.fields['glucose_mm'] = np.clip(self.fields['glucose_mm'], 0.0, 30.0)
            self.fields['lactate_mm'] = np.clip(self.fields['lactate_mm'], 0.0, 30.0)
            self.fields['temperature_c'] = np.clip(self.fields['temperature_c'], 35.0, 43.0)
            self.fields['ros_um'] = np.clip(self.fields['ros_um'], 0.0, 20.0)
            self.fields['glutamine_mm'] = np.clip(self.fields['glutamine_mm'], 0.0, 5.0)
            self.fields['calcium_um'] = np.clip(self.fields['calcium_um'], 50.0, 1000.0)
            self.fields['atp_adp_ratio'] = np.clip(self.fields['atp_adp_ratio'], 0.1, 20.0)
            self.fields['cytokine_pg_ml'] = np.clip(self.fields['cytokine_pg_ml'], 0.0, 200.0)

    def calculate_cancer_progression_score(self) -> float:
        """
        Calculate cancer progression score (0-100) from current field values
        Same algorithm as cancer_biology_simulator.py
        """
        score = 0.0

        # pH: Lower is worse
        if self.fields['ph_level'] < 7.0:
            score += (7.4 - self.fields['ph_level']) * 20

        # Oxygen: Lower is worse
        if self.fields['oxygen_percent'] < 10:
            score += (21 - self.fields['oxygen_percent']) * 2

        # Glucose: Higher is worse
        if self.fields['glucose_mm'] > 7.0:
            score += (self.fields['glucose_mm'] - 5.5) * 3

        # Lactate: Higher is worse
        if self.fields['lactate_mm'] > 2.0:
            score += (self.fields['lactate_mm'] - 1.0) * 4

        # ROS: Moderate increase
        if self.fields['ros_um'] > 3.0:
            score += (self.fields['ros_um'] - 0.1) * 2

        # Glutamine: Higher is worse
        if self.fields['glutamine_mm'] > 1.0:
            score += (self.fields['glutamine_mm'] - 0.6) * 8

        # Calcium: Dysregulation
        if abs(self.fields['calcium_um'] - 100) > 200:
            score += abs(self.fields['calcium_um'] - 100) / 20

        # ATP/ADP: Lower is worse
        if self.fields['atp_adp_ratio'] < 8:
            score += (10 - self.fields['atp_adp_ratio']) * 3

        # Cytokines: Higher is worse
        if self.fields['cytokine_pg_ml'] > 10:
            score += (self.fields['cytokine_pg_ml'] - 5.0) * 1.5

        return min(100, max(0, score))

    def calculate_metabolic_stress(self) -> float:
        """
        Calculate overall metabolic stress on cancer cells (0-1)
        Higher = more stress on cancer cells
        """
        stress = 0.0

        # Glucose restriction stress
        if self.fields['glucose_mm'] < 3.0:
            stress += (3.0 - self.fields['glucose_mm']) / 3.0 * 0.2

        # Glutamine restriction stress
        if self.fields['glutamine_mm'] < 0.4:
            stress += (0.4 - self.fields['glutamine_mm']) / 0.4 * 0.15

        # Hypoxia (paradoxically, normoxia stresses cancer cells adapted to hypoxia)
        if self.fields['oxygen_percent'] > 15:
            stress += (self.fields['oxygen_percent'] - 1.0) / 20.0 * 0.1

        # pH normalization stress
        if self.fields['ph_level'] > 7.2:
            stress += (self.fields['ph_level'] - 6.7) / 0.7 * 0.15

        # Hyperthermia stress
        if self.fields['temperature_c'] > 38:
            stress += (self.fields['temperature_c'] - 37) / 5.0 * 0.2

        # ROS stress (oxidative burst)
        if self.fields['ros_um'] > 3.0:
            stress += (self.fields['ros_um'] - 0.1) / 10.0 * 0.2

        return min(1.0, stress)

    def step(self, protocol: FieldInterventionProtocol, dt: float):
        """
        Advance time and apply protocol interventions
        """
        self.time += dt

        # Get active interventions
        active_interventions = protocol.get_active_interventions(self.time)

        # Apply effects
        self.apply_intervention_effects(active_interventions, dt)

    def record_response(self, tumor_cell_count: int, tumor_viability: float) -> InterventionResponse:
        """Record current state as response measurement"""
        response = InterventionResponse(
            time_hours=self.time,
            field_values=self.fields.copy(),
            tumor_cell_count=tumor_cell_count,
            tumor_viability=tumor_viability,
            metabolic_stress=self.calculate_metabolic_stress(),
            intervention_efficacy=1.0 - (self.calculate_cancer_progression_score() / 100.0),
        )
        self.history.append(response)
        return response


# ============================================================================
# PRE-DEFINED INTERVENTION PROTOCOLS (from ECH0 analysis)
# ============================================================================

def create_ech0_three_stage_protocol() -> FieldInterventionProtocol:
    """
    ECH0's recommended 3-stage multi-field intervention protocol
    From cancer remission analysis
    """
    protocol = FieldInterventionProtocol(
        name="ECH0 Three-Stage Protocol",
        description="Simultaneous multi-field targeting based on ECH0 14B analysis",
    )

    # Stage 1: Metabolic Stress & Immunosuppression (Days 1-7)
    ketogenic_diet = FieldIntervention(
        name="Ketogenic Diet",
        intervention_type=InterventionType.DIETARY,
        target_fields=['glucose_mm', 'glutamine_mm', 'atp_adp_ratio'],
        effects={
            'glucose_mm': -0.5,  # Gradual reduction per hour
            'glutamine_mm': -0.04,
            'atp_adp_ratio': +0.02,
        },
        onset_hours=24.0,
        duration_hours=1000.0,  # Sustained
        intensity=1.0,
    )

    hbot = FieldIntervention(
        name="Hyperbaric Oxygen",
        intervention_type=InterventionType.PHYSICAL,
        target_fields=['oxygen_percent'],
        effects={'oxygen_percent': +0.5},  # Gradual normalization
        onset_hours=2.0,
        duration_hours=1000.0,
        intensity=1.0,
    )

    hyperthermia = FieldIntervention(
        name="Mild Hyperthermia",
        intervention_type=InterventionType.PHYSICAL,
        target_fields=['temperature_c'],
        effects={'temperature_c': +0.1},
        onset_hours=0.5,
        duration_hours=1000.0,
        intensity=1.0,
    )

    immunotherapy = FieldIntervention(
        name="Immunotherapy",
        intervention_type=InterventionType.PHARMACOLOGICAL,
        target_fields=['cytokine_pg_ml'],
        effects={'cytokine_pg_ml': -2.0},
        onset_hours=12.0,
        duration_hours=1000.0,
        intensity=1.0,
    )

    # Add Stage 1 interventions (start immediately)
    protocol.add_intervention(ketogenic_diet, start_time=0.0)
    protocol.add_intervention(hbot, start_time=0.0)
    protocol.add_intervention(hyperthermia, start_time=0.0)
    protocol.add_intervention(immunotherapy, start_time=0.0)

    # Stage 2: DNA Damage & Apoptosis (Day 7+)
    # These would be actual drugs in practice; here we simulate their field effects
    oxidative_burst = FieldIntervention(
        name="Oxidative Therapy",
        intervention_type=InterventionType.PHARMACOLOGICAL,
        target_fields=['ros_um'],
        effects={'ros_um': +0.2},
        onset_hours=2.0,
        duration_hours=1000.0,
        intensity=1.0,
    )

    protocol.add_intervention(oxidative_burst, start_time=7 * 24.0)  # Day 7

    # Stage 3: Microenvironment Disruption (Day 21+)
    ph_normalization = FieldIntervention(
        name="pH Normalization",
        intervention_type=InterventionType.PHARMACOLOGICAL,
        target_fields=['ph_level', 'lactate_mm'],
        effects={
            'ph_level': +0.03,
            'lactate_mm': -0.4,
        },
        onset_hours=12.0,
        duration_hours=1000.0,
        intensity=1.0,
    )

    protocol.add_intervention(ph_normalization, start_time=21 * 24.0)  # Day 21

    return protocol


def create_standard_chemotherapy_protocol() -> FieldInterventionProtocol:
    """
    Standard chemotherapy (for comparison)
    Less comprehensive field targeting
    """
    protocol = FieldInterventionProtocol(
        name="Standard Chemotherapy",
        description="Traditional chemotherapy without metabolic support",
    )

    # Chemotherapy affects cancer cells but doesn't optimize the field
    chemo = FieldIntervention(
        name="Chemotherapy",
        intervention_type=InterventionType.PHARMACOLOGICAL,
        target_fields=['cytokine_pg_ml', 'ros_um'],  # Inflammation + oxidative stress
        effects={
            'cytokine_pg_ml': +10.0,  # Increases inflammation
            'ros_um': +1.0,
        },
        onset_hours=1.0,
        duration_hours=72.0,  # 3 days
        intensity=1.0,
    )

    # Cycle every 21 days
    for cycle in range(4):
        protocol.add_intervention(chemo, start_time=cycle * 21 * 24.0)

    return protocol
