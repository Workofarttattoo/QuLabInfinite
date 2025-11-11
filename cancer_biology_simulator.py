"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Cancer Biology Simulator - QuLabInfinite Integration
Explores cellular signaling pathways and metabolic conditions for cancer remission
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from chemistry_lab import (
    ChemistryLaboratory, ChemistryLabConfig,
    ReactionSimulator, MolecularDynamics,
    QuantumChemistryInterface, QMMethod
)
import json


class CancerCellEnvironment:
    """
    Simulates the microenvironment of cancer cells with 10 key fields
    that influence cancer progression and potential remission.
    """

    def __init__(self):
        self.fields = {
            # 1. pH Level - Cancer cells thrive in acidic environments (pH 6.5-7.0)
            'ph_level': 7.4,  # Normal tissue pH

            # 2. Oxygen Concentration - Hypoxia promotes cancer (normal ~21% O2)
            'oxygen_percent': 21.0,

            # 3. Glucose Concentration - Warburg effect (mM)
            'glucose_mm': 5.5,  # Normal blood glucose

            # 4. Lactate Levels - Metabolic waste product (mM)
            'lactate_mm': 1.0,  # Normal levels

            # 5. Temperature - Hyperthermia effects (Celsius)
            'temperature_c': 37.0,  # Normal body temperature

            # 6. Reactive Oxygen Species (ROS) - μM H2O2 equivalent
            'ros_um': 0.1,  # Normal oxidative stress

            # 7. Glutamine Concentration - Energy metabolism (mM)
            'glutamine_mm': 0.6,  # Normal plasma levels

            # 8. Calcium Ion Concentration - Signaling (μM)
            'calcium_um': 100.0,  # Resting intracellular Ca2+

            # 9. ATP/ADP Ratio - Energy state
            'atp_adp_ratio': 10.0,  # Healthy cell ratio

            # 10. Pro-inflammatory Cytokines - Immune signaling (pg/mL)
            'cytokine_pg_ml': 5.0,  # Normal IL-6 levels
        }

        # Cancer-promoting conditions
        self.cancer_conditions = {
            'ph_level': 6.7,  # Acidic
            'oxygen_percent': 1.0,  # Severe hypoxia
            'glucose_mm': 15.0,  # High glucose (Warburg)
            'lactate_mm': 10.0,  # High lactate
            'temperature_c': 37.0,
            'ros_um': 5.0,  # Oxidative stress
            'glutamine_mm': 2.0,  # High glutamine
            'calcium_um': 500.0,  # Elevated Ca2+
            'atp_adp_ratio': 5.0,  # Lower energy
            'cytokine_pg_ml': 50.0,  # Inflammation
        }

        # Remission-promoting conditions (hypothetical optimal)
        self.remission_conditions = {
            'ph_level': 7.4,  # Normalized pH
            'oxygen_percent': 21.0,  # Normoxia
            'glucose_mm': 4.0,  # Lower glucose (ketogenic?)
            'lactate_mm': 0.5,  # Low lactate
            'temperature_c': 39.0,  # Mild hyperthermia
            'ros_um': 2.0,  # Moderate oxidative burst
            'glutamine_mm': 0.2,  # Restricted glutamine
            'calcium_um': 150.0,  # Moderate elevation
            'atp_adp_ratio': 12.0,  # High energy
            'cytokine_pg_ml': 2.0,  # Low inflammation
        }

        self.chem_lab = ChemistryLaboratory(ChemistryLabConfig())

    def set_conditions(self, condition_type='normal'):
        """Set environment to normal, cancer, or remission conditions."""
        if condition_type == 'cancer':
            self.fields = self.cancer_conditions.copy()
        elif condition_type == 'remission':
            self.fields = self.remission_conditions.copy()
        else:
            # Reset to normal
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

    def calculate_cancer_progression_score(self):
        """
        Calculate how favorable the environment is for cancer (0-100).
        Higher score = more cancer-promoting.
        """
        score = 0

        # pH: Lower is worse (acidic promotes cancer)
        if self.fields['ph_level'] < 7.0:
            score += (7.4 - self.fields['ph_level']) * 20

        # Oxygen: Lower is worse (hypoxia)
        if self.fields['oxygen_percent'] < 10:
            score += (21 - self.fields['oxygen_percent']) * 2

        # Glucose: Higher is worse (Warburg effect)
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

        # Cytokines: Higher is worse (inflammation)
        if self.fields['cytokine_pg_ml'] > 10:
            score += (self.fields['cytokine_pg_ml'] - 5.0) * 1.5

        return min(100, max(0, score))

    def simulate_metabolic_pathway(self, pathway='glycolysis'):
        """Simulate key metabolic pathways in current conditions."""
        results = {}

        if pathway == 'glycolysis':
            # Glucose -> Pyruvate -> Lactate (Warburg effect)
            glucose_flux = self.fields['glucose_mm'] * (1.0 if self.fields['oxygen_percent'] < 5 else 0.3)
            lactate_production = glucose_flux * 2.0  # 2 lactate per glucose

            results = {
                'glucose_consumption_rate': glucose_flux,
                'lactate_production_rate': lactate_production,
                'atp_production': glucose_flux * 2,  # Only 2 ATP per glucose in glycolysis
                'pathway_efficiency': 'low_aerobic' if self.fields['oxygen_percent'] < 5 else 'normal'
            }

        elif pathway == 'oxidative_phosphorylation':
            # Oxygen-dependent ATP production
            if self.fields['oxygen_percent'] > 10:
                atp_production = self.fields['oxygen_percent'] * 1.8
                results = {
                    'atp_production': atp_production,
                    'efficiency': 'high',
                    'ros_generation': atp_production * 0.02
                }
            else:
                results = {
                    'atp_production': 0.5,
                    'efficiency': 'minimal',
                    'ros_generation': 0.1
                }

        elif pathway == 'glutaminolysis':
            # Glutamine -> Glutamate -> α-ketoglutarate (cancer fuel)
            glutamine_flux = self.fields['glutamine_mm'] * 1.5
            results = {
                'glutamine_consumption': glutamine_flux,
                'atp_production': glutamine_flux * 3,
                'biosynthesis_support': glutamine_flux * 2
            }

        return results

    def apply_intervention(self, intervention_name, intensity=1.0):
        """
        Apply therapeutic interventions to shift conditions toward remission.
        """
        interventions = {
            'alkaline_diet': {
                'ph_level': +0.3 * intensity,
            },
            'hyperbaric_oxygen': {
                'oxygen_percent': +15.0 * intensity,
            },
            'ketogenic_diet': {
                'glucose_mm': -3.0 * intensity,
                'glutamine_mm': -0.4 * intensity,
            },
            'hyperthermia': {
                'temperature_c': +2.0 * intensity,
            },
            'antioxidant_therapy': {
                'ros_um': -2.0 * intensity,
            },
            'immunotherapy': {
                'cytokine_pg_ml': -20.0 * intensity,
            },
            'metabolic_therapy': {
                'glucose_mm': -4.0 * intensity,
                'glutamine_mm': -0.5 * intensity,
                'atp_adp_ratio': +3.0 * intensity,
            },
        }

        if intervention_name in interventions:
            changes = interventions[intervention_name]
            for field, delta in changes.items():
                self.fields[field] = max(0, self.fields[field] + delta)
            return True
        return False

    def get_report(self):
        """Generate detailed report of current cellular environment."""
        progression_score = self.calculate_cancer_progression_score()

        report = {
            'current_conditions': self.fields.copy(),
            'cancer_progression_score': progression_score,
            'interpretation': self._interpret_score(progression_score),
            'metabolic_pathways': {
                'glycolysis': self.simulate_metabolic_pathway('glycolysis'),
                'oxidative_phosphorylation': self.simulate_metabolic_pathway('oxidative_phosphorylation'),
                'glutaminolysis': self.simulate_metabolic_pathway('glutaminolysis'),
            },
            'recommendations': self._generate_recommendations(progression_score)
        }

        return report

    def _interpret_score(self, score):
        """Interpret the cancer progression score."""
        if score < 20:
            return "Healthy cellular environment - low cancer risk"
        elif score < 40:
            return "Mild cancer-promoting conditions detected"
        elif score < 60:
            return "Moderate cancer-promoting environment"
        elif score < 80:
            return "High cancer progression risk"
        else:
            return "Severe cancer-promoting conditions"

    def _generate_recommendations(self, score):
        """Generate intervention recommendations based on score."""
        recommendations = []

        if self.fields['ph_level'] < 7.2:
            recommendations.append("Consider alkaline diet to normalize pH")

        if self.fields['oxygen_percent'] < 15:
            recommendations.append("Hyperbaric oxygen therapy may help")

        if self.fields['glucose_mm'] > 8:
            recommendations.append("Ketogenic diet to reduce glucose availability")

        if self.fields['lactate_mm'] > 5:
            recommendations.append("Metabolic therapy to reduce lactate buildup")

        if self.fields['ros_um'] > 3:
            recommendations.append("Targeted oxidative therapy (paradoxical ROS burst)")

        if self.fields['glutamine_mm'] > 1.0:
            recommendations.append("Glutamine restriction therapy")

        if self.fields['cytokine_pg_ml'] > 20:
            recommendations.append("Anti-inflammatory immunotherapy")

        if score > 60:
            recommendations.append("Combination metabolic therapy strongly recommended")

        return recommendations


def main():
    """Run cancer biology simulation."""
    print("=" * 80)
    print("Cancer Biology Simulator - QuLabInfinite")
    print("Exploring cellular signaling and metabolic pathways")
    print("=" * 80)
    print()

    env = CancerCellEnvironment()

    # Test 1: Normal conditions
    print("### NORMAL TISSUE CONDITIONS ###")
    env.set_conditions('normal')
    report = env.get_report()
    print(json.dumps(report, indent=2))
    print()

    # Test 2: Cancer conditions
    print("\n### CANCER MICROENVIRONMENT CONDITIONS ###")
    env.set_conditions('cancer')
    report = env.get_report()
    print(json.dumps(report, indent=2))
    print()

    # Test 3: Apply interventions
    print("\n### APPLYING THERAPEUTIC INTERVENTIONS ###")
    env.set_conditions('cancer')
    print("Starting cancer progression score:", env.calculate_cancer_progression_score())

    interventions = ['ketogenic_diet', 'hyperbaric_oxygen', 'hyperthermia', 'immunotherapy']
    for intervention in interventions:
        env.apply_intervention(intervention, intensity=0.8)
        print(f"After {intervention}: Score = {env.calculate_cancer_progression_score():.1f}")

    print("\nFinal report after interventions:")
    final_report = env.get_report()
    print(json.dumps(final_report, indent=2))

    # Test 4: Remission conditions
    print("\n### OPTIMAL REMISSION CONDITIONS ###")
    env.set_conditions('remission')
    report = env.get_report()
    print(json.dumps(report, indent=2))

    print("\n" + "=" * 80)
    print("10 KEY FIELDS FOR CANCER BIOLOGY:")
    print("=" * 80)
    for i, (field, value) in enumerate(env.fields.items(), 1):
        print(f"{i:2d}. {field:20s} = {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
