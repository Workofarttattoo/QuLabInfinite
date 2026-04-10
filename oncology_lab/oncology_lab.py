"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

oncology_lab - Part of Oncology Lab
"""

"""
QuLabInfinite Oncology Laboratory Interface.

This module wires together the tumor-growth, drug-response, and intervention
subsystems to provide a programmable sandbox for oncology-inspired experiments.
It is a research prototype that uses heuristic parameters assembled from open
literature and reasonable defaults. The simulator is not a substitute for
clinical data, medical judgment, or laboratory validation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

from .tumor_simulator import (
    TumorSimulator, TumorGrowthModel, CancerCell, CellCyclePhase
)
from .drug_response import (
    DrugSimulator, Drug, get_drug_from_database, list_available_drugs
)
from .ten_field_controller import (
    TenFieldController, FieldInterventionProtocol,
    create_ech0_three_stage_protocol, create_standard_chemotherapy_protocol
)


class TumorType(Enum):
    """Common tumor types with specific characteristics"""
    BREAST_CANCER = "breast_cancer"
    LUNG_CANCER = "lung_cancer"
    COLORECTAL_CANCER = "colorectal_cancer"
    PROSTATE_CANCER = "prostate_cancer"
    PANCREATIC_CANCER = "pancreatic_cancer"
    GLIOBLASTOMA = "glioblastoma"
    MELANOMA = "melanoma"
    OVARIAN_CANCER = "ovarian_cancer"


class CancerStage(Enum):
    """Cancer staging"""
    STAGE_I = 1
    STAGE_II = 2
    STAGE_III = 3
    STAGE_IV = 4  # Metastatic


@dataclass
class OncologyLabConfig:
    """Configuration for oncology lab experiments"""
    tumor_type: TumorType = TumorType.BREAST_CANCER
    stage: CancerStage = CancerStage.STAGE_II
    initial_tumor_cells: int = 1000
    growth_model: TumorGrowthModel = TumorGrowthModel.GOMPERTZIAN
    enable_microenvironment: bool = True
    enable_immune_system: bool = False  # Future feature
    time_step_hours: float = 0.1


class OncologyLaboratory:
    """
    Main laboratory for conducting cancer experiments
    Integrates tumor simulation, drug testing, and intervention protocols
    """

    def __init__(self, config: Optional[OncologyLabConfig] = None):
        """
        Initialize the oncology laboratory

        Args:
            config: Laboratory configuration
        """
        self.config = config or OncologyLabConfig()

        # Initialize tumor simulator
        self.tumor = TumorSimulator(
            tumor_type=self.config.tumor_type.value,
            growth_model=self.config.growth_model,
            initial_cells=self.config.initial_tumor_cells
        )

        # Initialize field controller
        self.field_controller = TenFieldController()
        profile = self._derive_tumor_profile(self.config.tumor_type, self.config.stage)
        self.field_controller.apply_cancer_profile(profile.get('microenvironment'))
        self.field_controller.set_cancer_microenvironment()
        self.tumor.intrinsic_growth_rate = profile['growth_rate']
        self.tumor.carrying_capacity = profile['carrying_capacity']
        self.tumor.gompertz_retardation = profile['gompertz_retardation']
        self.tumor.apply_field_overrides(self.field_controller.fields)
        self.drug_sensitivity = profile['drug_sensitivity']
        self.profile = profile

        # Drug simulators
        self.active_drugs: Dict[str, DrugSimulator] = {}

        # Experimental results
        self.experiment_log = []
        self.time = 0.0

        print("[OncologyLab] Laboratory initialized")
        print(f"  Tumor type: {self.config.tumor_type.value}")
        print(f"  Stage: {self.config.stage.value}")
        print(f"  Initial cells: {self.config.initial_tumor_cells}")
        print(f"  Growth model: {self.config.growth_model.value}")

    def administer_drug(self, drug_name: str, dose_mg: float):
        """
        Administer a drug to the tumor

        Args:
            drug_name: Name of drug from database
            dose_mg: Dose in milligrams
        """
        drug = get_drug_from_database(drug_name)
        if drug is None:
            available = list_available_drugs()
            raise ValueError(f"Drug '{drug_name}' not found. Available: {available}")

        if drug_name not in self.active_drugs:
            self.active_drugs[drug_name] = DrugSimulator(drug)

        self.active_drugs[drug_name].administer_dose(dose_mg, self.time)
        print(f"[OncologyLab] Administered {dose_mg} mg of {drug_name} at t={self.time:.1f}h")

    def apply_intervention_protocol(self, protocol: FieldInterventionProtocol):
        """
        Apply a multi-field intervention protocol

        Args:
            protocol: Intervention protocol to apply
        """
        print(f"[OncologyLab] Applying protocol: {protocol.name}")
        print(f"  Description: {protocol.description}")
        print(f"  Interventions: {len(protocol.interventions)}")
        self.active_protocol = protocol

    def step(self, dt: Optional[float] = None):
        """
        Advance simulation by one time step

        Args:
            dt: Time step in hours (default: from config)
        """
        if dt is None:
            dt = self.config.time_step_hours

        # Update field controller with active protocol
        if hasattr(self, 'active_protocol'):
            self.field_controller.step(self.active_protocol, dt)
        else:
            # Progress time even without a protocol so measurements align
            self.field_controller.time += dt

        # Tie controller output into tumor microenvironment
        self.tumor.apply_field_overrides(self.field_controller.fields)

        # Update tumor microenvironment from fields
        for cell in self.tumor.cells:
            if not cell.is_alive:
                continue

            # Apply drug concentrations
            for drug_name, drug_sim in self.active_drugs.items():
                tumor_conc = drug_sim.get_tumor_concentration(self.time)
                cell.drug_concentrations[drug_name] = tumor_conc

                # Calculate drug-induced death probability
                drug = drug_sim.drug
                death_prob = drug.calculate_cell_kill_probability(tumor_conc, dt)
                death_prob = np.clip(death_prob * self.drug_sensitivity, 0.0, 1.0)

                # Apply drug effect
                if np.random.random() < death_prob:
                    cell.is_alive = False
                    cell.phase = CellCyclePhase.APOPTOSIS

                    # Ensure dead-cell pruning can detect recent deaths
                    if hasattr(cell, 'time_since_death'):
                        cell.time_since_death = 0.0

        # Advance tumor simulation
        self.tumor.step(dt)

        # Record response
        stats = self.tumor.get_statistics()
        self.field_controller.record_response(
            tumor_cell_count=stats['alive_cells'],
            tumor_viability=stats['average_viability']
        )

        # Update time
        self.time += dt

    def run_experiment(self, duration_days: float, report_interval_hours: float = 24.0):
        """
        Run experiment for specified duration

        Args:
            duration_days: How long to run (days)
            report_interval_hours: How often to print progress
        """
        print(f"\n[OncologyLab] Starting experiment: {duration_days} days")
        print("=" * 70)

        duration_hours = duration_days * 24.0
        steps = int(duration_hours / self.config.time_step_hours)
        last_report_time = 0.0

        for _ in range(steps):
            self.step()

            if self.time - last_report_time >= report_interval_hours:
                stats = self.tumor.get_statistics()
                progression_score = self.field_controller.calculate_cancer_progression_score()

                print(f"\n[Day {self.time/24:.1f}]")
                print(f"  Tumor cells: {stats['alive_cells']:,}")
                print(f"  Volume: {stats['tumor_volume_mm3']:.2f} mm³")
                print(f"  Viability: {stats['average_viability']:.2%}")
                print(f"  Cancer progression score: {progression_score:.1f}/100")
                print(f"  Metabolic stress: {self.field_controller.calculate_metabolic_stress():.2%}")

                last_report_time = self.time

        print("\n" + "=" * 70)
        print("[OncologyLab] Experiment complete")

    def _derive_tumor_profile(self, tumor_type: TumorType, stage: CancerStage) -> Dict:
        """
        Map tumour type and stage to growth dynamics, drug sensitivity, and
        preferred cancer-field baselines so experiments respond to config flags.
        """
        import json
        import os

        profiles_path = os.path.join(os.path.dirname(__file__), 'tumor_profiles.json')
        try:
            with open(profiles_path, 'r') as f:
                all_profiles = json.load(f)
        except Exception as e:
            # Fallback for testing/if file is missing
            print(f"Warning: Could not load tumor_profiles.json: {e}")
            return {
                'growth_rate': 0.03,
                'gompertz_retardation': 0.001,
                'carrying_capacity': 1e9,
                'drug_sensitivity': 1.0,
                'microenvironment': {},
            }

        type_name = tumor_type.name
        stage_name = stage.name

        if type_name in all_profiles and stage_name in all_profiles[type_name]:
            return all_profiles[type_name][stage_name].copy()

        print(f"Warning: Profile not found for {type_name} {stage_name}")
        return {
            'growth_rate': 0.03,
            'gompertz_retardation': 0.001,
            'carrying_capacity': 1e9,
            'drug_sensitivity': 1.0,
            'microenvironment': {},
        }


        duration_hours = duration_days * 24.0
        steps = int(duration_hours / self.config.time_step_hours)
        last_report_time = 0.0

        for i in range(steps):
            self.step()

            # Periodic reporting
            if self.time - last_report_time >= report_interval_hours:
                stats = self.tumor.get_statistics()
                progression_score = self.field_controller.calculate_cancer_progression_score()

                print(f"\n[Day {self.time/24:.1f}]")
                print(f"  Tumor cells: {stats['alive_cells']:,}")
                print(f"  Volume: {stats['tumor_volume_mm3']:.2f} mm³")
                print(f"  Viability: {stats['average_viability']:.2%}")
                print(f"  Cancer progression score: {progression_score:.1f}/100")
                print(f"  Metabolic stress: {self.field_controller.calculate_metabolic_stress():.2%}")

                last_report_time = self.time

        print("\n" + "=" * 70)
        print("[OncologyLab] Experiment complete")

    def get_results(self) -> Dict:
        """
        Get comprehensive experimental results

        Returns:
            Dictionary with all experimental data
        """
        tumor_stats = self.tumor.get_statistics()
        field_history = self.field_controller.history

        # Extract time series
        time_points = [r.time_hours for r in field_history]
        cell_counts = [r.tumor_cell_count for r in field_history]
        viabilities = [r.tumor_viability for r in field_history]
        progression_scores = [
            self._calculate_score_from_fields(r.field_values)
            for r in field_history
        ]

        # Drug concentrations over time
        drug_data = {}
        for drug_name, drug_sim in self.active_drugs.items():
            drug_data[drug_name] = {
                'plasma_conc': [drug_sim.get_plasma_concentration(t) for t in time_points],
                'tumor_conc': [drug_sim.get_tumor_concentration(t) for t in time_points],
                'effect': [drug_sim.get_effect(t) for t in time_points],
            }

        # Field values over time
        field_data = {}
        for field_name in self.field_controller.fields.keys():
            field_data[field_name] = [r.field_values[field_name] for r in field_history]

        return {
            'final_stats': tumor_stats,
            'time_hours': time_points,
            'time_days': [t/24 for t in time_points],
            'cell_counts': cell_counts,
            'viabilities': viabilities,
            'progression_scores': progression_scores,
            'field_data': field_data,
            'drug_data': drug_data,
            'config': {
                'tumor_type': self.config.tumor_type.value,
                'stage': self.config.stage.value,
                'initial_cells': self.config.initial_tumor_cells,
                'growth_model': self.config.growth_model.value,
            }
        }

    def _calculate_score_from_fields(self, fields: Dict[str, float]) -> float:
        """Helper to calculate progression score from field dict"""
        # Temporarily set fields
        old_fields = self.field_controller.fields.copy()
        self.field_controller.fields = fields
        score = self.field_controller.calculate_cancer_progression_score()
        self.field_controller.fields = old_fields
        return score

    def print_summary(self):
        """Print experimental summary"""
        stats = self.tumor.get_statistics()
        final_score = self.field_controller.calculate_cancer_progression_score()

        print("\n" + "=" * 70)
        print("EXPERIMENTAL SUMMARY")
        print("=" * 70)
        print(f"Duration: {self.time/24:.1f} days ({self.time:.1f} hours)")
        print(f"\nTumor Status:")
        print(f"  Total cells: {stats['total_cells']:,}")
        print(f"  Alive cells: {stats['alive_cells']:,}")
        print(f"  Dead cells: {stats['dead_cells']:,}")
        print(f"  Tumor volume: {stats['tumor_volume_mm3']:.2f} mm³")
        print(f"  Average viability: {stats['average_viability']:.2%}")

        print(f"\nMicroenvironment (10 Fields):")
        for field_name, value in self.field_controller.fields.items():
            baseline = self.field_controller.baseline_fields[field_name]
            cancer = self.field_controller.cancer_fields[field_name]
            print(f"  {field_name:20s}: {value:8.2f} (baseline: {baseline:.2f}, cancer: {cancer:.2f})")

        print(f"\nCancer Progression Score: {final_score:.1f}/100")
        print(f"Metabolic Stress: {self.field_controller.calculate_metabolic_stress():.2%}")

        if self.active_drugs:
            print(f"\nDrugs Administered:")
            for drug_name in self.active_drugs:
                print(f"  - {drug_name}")

        print("=" * 70)
