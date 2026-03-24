import logging
#!/usr/bin/env python3
"""
ECH0 Digital Twin Characterizer
Creates and simulates digital twins of metamaterials for comprehensive characterization

ECH0 Capabilities:
- Digital twin creation from metamaterial designs
- Multi-physics simulation (electromagnetic, acoustic, mechanical)
- Performance characterization under various conditions
- Failure mode analysis and optimization recommendations
- Real-time parameter optimization

Digital Twin Features:
- Virtual prototyping before physical fabrication
- Predictive maintenance and performance monitoring
- Multi-scale simulation (unit cell to bulk material)
- Environmental condition testing
- Cost-performance optimization
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import random
import copy
import math
from pathlib import Path

from ech0_interface import ECH0_QuLabInterface


@dataclass
class DigitalTwin:
    """Complete digital twin of a metamaterial"""
    name: str
    original_design: Dict[str, Any]
    twin_id: str
    creation_timestamp: str
    characterization_results: Dict[str, Any] = field(default_factory=dict)
    simulation_history: List[Dict[str, Any]] = field(default_factory=list)
    optimization_recommendations: List[str] = field(default_factory=list)
    failure_modes_identified: List[Dict[str, Any]] = field(default_factory=list)
    confidence_level: float = 0.0
    validation_status: str = "created"


@dataclass
class CharacterizationCondition:
    """Environmental/operational conditions for digital twin testing"""
    name: str
    temperature_range: Tuple[float, float]  # Kelvin
    pressure_range: Tuple[float, float]  # Pa
    frequency_range: Optional[Tuple[float, float]] = None  # Hz (for EM/acoustic)
    strain_range: Optional[Tuple[float, float]] = None  # dimensionless (for mechanical)
    humidity_range: Optional[Tuple[float, float]] = None  # %
    magnetic_field: Optional[float] = None  # Tesla
    electric_field: Optional[float] = None  # V/m


class ECH0_DigitalTwinCharacterizer:
    """
    ECH0-driven digital twin characterization system

    Creates virtual representations of metamaterials and simulates their
    behavior under various conditions to predict real-world performance.
    """

    def __init__(self):
        self.interface = ECH0_QuLabInterface()
        self.digital_twins: Dict[str, DigitalTwin] = {}
        self.characterization_conditions = self._initialize_conditions()

        # ECH0 knowledge base for digital twin simulation
        self.material_models = {
            'copper': {
                'conductivity': 5.96e7,  # S/m at 20°C
                'thermal_conductivity': 401,  # W/(m·K)
                'youngs_modulus': 110e9,  # Pa
                'poisson_ratio': 0.34,
                'density': 8960,  # kg/m³
                'temperature_coefficient': 0.0039  # 1/K
            },
            'silicon_dioxide': {
                'conductivity': 1e-14,  # S/m
                'thermal_conductivity': 1.4,  # W/(m·K)
                'youngs_modulus': 70e9,  # Pa
                'poisson_ratio': 0.17,
                'density': 2650,  # kg/m³
                'refractive_index': 1.45
            },
            'aluminum': {
                'conductivity': 3.5e7,  # S/m
                'thermal_conductivity': 237,  # W/(m·K)
                'youngs_modulus': 69e9,  # Pa
                'poisson_ratio': 0.33,
                'density': 2700,  # kg/m³
                'temperature_coefficient': 0.0043  # 1/K
            },
            'polystyrene': {
                'thermal_conductivity': 0.13,  # W/(m·K)
                'youngs_modulus': 3.0e9,  # Pa
                'poisson_ratio': 0.35,
                'density': 1050,  # kg/m³
                'acoustic_impedance': 2.4e6,  # kg/(m²·s)
                'sound_velocity': 2350  # m/s
            }
        }

    def _initialize_conditions(self) -> Dict[str, CharacterizationCondition]:
        """Initialize comprehensive characterization conditions with extensive stress factors"""

        return {
            # Basic Environmental Conditions
            'standard_lab': CharacterizationCondition(
                name='Standard Laboratory Conditions (Baseline)',
                temperature_range=(293, 298),  # 20-25°C
                pressure_range=(101325, 101325),  # 1 atm
                frequency_range=(1e6, 1e11),  # 1 MHz - 100 GHz
                strain_range=(-0.01, 0.01),  # ±1% strain
                humidity_range=(40, 60)  # 40-60% RH
            ),

            # Temperature Stress Factors
            'extreme_temperature': CharacterizationCondition(
                name='Extreme Temperature Cycling (-100°C to 100°C)',
                temperature_range=(173, 373),  # -100°C to 100°C
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.05, 0.05)
            ),
            'cryogenic_temperature': CharacterizationCondition(
                name='Cryogenic Temperature (-269°C to -150°C)',
                temperature_range=(4, 123),  # 4K to 123K (liquid helium to liquid nitrogen)
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e12),  # Extended for quantum effects
                strain_range=(-0.02, 0.02),
                magnetic_field=1.0  # 1 Tesla (typical for cryogenic systems)
            ),
            'high_temperature': CharacterizationCondition(
                name='High Temperature Furnace (100°C to 1000°C)',
                temperature_range=(373, 1273),  # 100°C to 1000°C
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.08, 0.08),
                humidity_range=(10, 30)  # Low humidity in furnace
            ),

            # Pressure Stress Factors
            'high_pressure': CharacterizationCondition(
                name='High Pressure Hydrostatic (1 atm to 100 MPa)',
                temperature_range=(293, 298),
                pressure_range=(101325, 1e7),  # 1 atm to 100 MPa
                frequency_range=(1e6, 1e11),
                strain_range=(-0.1, 0.1)
            ),
            'ultra_high_pressure': CharacterizationCondition(
                name='Ultra-High Pressure Diamond Anvil (100 MPa to 10 GPa)',
                temperature_range=(293, 298),
                pressure_range=(1e7, 1e10),  # 100 MPa to 10 GPa
                frequency_range=(1e6, 1e12),  # Extended frequency range
                strain_range=(-0.5, 0.5),  # Large strains
                magnetic_field=5.0  # High magnetic field
            ),
            'vacuum_conditions': CharacterizationCondition(
                name='High Vacuum Environment (10^-6 to 10^-9 Torr)',
                temperature_range=(293, 298),
                pressure_range=(1e-4, 1e-7),  # 10^-6 to 10^-9 Torr
                frequency_range=(1e6, 1e12),
                strain_range=(-0.01, 0.01),
                humidity_range=(0, 0.1)  # Extremely dry
            ),

            # Corrosion and Chemical Stress Factors
            'saltwater_corrosion': CharacterizationCondition(
                name='Saltwater Corrosion Test (3.5% NaCl)',
                temperature_range=(293, 308),  # 20-35°C
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.02, 0.02),
                humidity_range=(80, 100),  # High humidity for corrosion
                electric_field=1e3  # 1000 V/m (galvanic corrosion)
            ),
            'acid_exposure': CharacterizationCondition(
                name='Acid Exposure Test (pH 2-4)',
                temperature_range=(293, 333),  # 20-60°C
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.03, 0.03),
                humidity_range=(60, 90)
            ),
            'alkaline_exposure': CharacterizationCondition(
                name='Alkaline Exposure Test (pH 10-12)',
                temperature_range=(293, 333),  # 20-60°C
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.03, 0.03),
                humidity_range=(60, 90)
            ),
            'oxidizing_atmosphere': CharacterizationCondition(
                name='Oxidizing Atmosphere (High Oxygen)',
                temperature_range=(293, 373),  # 20-100°C
                pressure_range=(101325, 2e5),  # Slightly elevated pressure
                frequency_range=(1e6, 1e11),
                strain_range=(-0.02, 0.02),
                humidity_range=(50, 80)
            ),

            # Mechanical Stress Factors
            'vibration_testing': CharacterizationCondition(
                name='Random Vibration Testing (10-2000 Hz)',
                temperature_range=(293, 298),
                pressure_range=(101325, 101325),
                frequency_range=(10, 2000),  # Hz for vibration
                strain_range=(-0.05, 0.05),
                humidity_range=(40, 60)
            ),
            'shock_testing': CharacterizationCondition(
                name='Mechanical Shock Testing',
                temperature_range=(293, 298),
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-1.0, 1.0),  # Large shock strains
                humidity_range=(40, 60)
            ),
            'fatigue_cycling': CharacterizationCondition(
                name='Fatigue Cycling Test (10^6 cycles)',
                temperature_range=(293, 298),
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.02, 0.02),
                humidity_range=(40, 60)
            ),

            # Radiation and Electromagnetic Stress Factors
            'gamma_radiation': CharacterizationCondition(
                name='Gamma Radiation Exposure (1-100 kGy)',
                temperature_range=(293, 298),
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.01, 0.01),
                humidity_range=(40, 60),
                magnetic_field=0.1  # Low magnetic field
            ),
            'neutron_radiation': CharacterizationCondition(
                name='Neutron Radiation Exposure (10^12-10^15 n/cm²)',
                temperature_range=(293, 298),
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.01, 0.01),
                humidity_range=(40, 60),
                magnetic_field=0.5  # Moderate magnetic field
            ),
            'uv_exposure': CharacterizationCondition(
                name='UV Radiation Exposure (UVA/UVB/UVC)',
                temperature_range=(293, 333),  # 20-60°C
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.01, 0.01),
                humidity_range=(30, 70)
            ),
            'electromagnetic_interference': CharacterizationCondition(
                name='High Electromagnetic Field Exposure',
                temperature_range=(293, 298),
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e12),  # Extended range
                strain_range=(-0.01, 0.01),
                humidity_range=(40, 60),
                magnetic_field=10.0,  # 10 Tesla
                electric_field=1e6  # 1 MV/m
            ),

            # Biological and Environmental Stress Factors
            'biodegradation': CharacterizationCondition(
                name='Biodegradation Test (Enzymatic/Microbial)',
                temperature_range=(298, 310),  # 25-37°C (body temperature)
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.01, 0.01),
                humidity_range=(90, 100)  # High humidity for microbial growth
            ),
            'humidity_cycling': CharacterizationCondition(
                name='Humidity Cycling (10% to 95% RH)',
                temperature_range=(293, 298),
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.02, 0.02),
                humidity_range=(10, 95)  # Full humidity range
            ),
            'thermal_shock': CharacterizationCondition(
                name='Thermal Shock Cycling (-55°C to 125°C)',
                temperature_range=(218, 398),  # -55°C to 125°C
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.08, 0.08),
                humidity_range=(20, 80)
            ),

            # Space and Aerospace Stress Factors
            'space_vacuum': CharacterizationCondition(
                name='Space Vacuum with Radiation',
                temperature_range=(173, 323),  # -100°C to 50°C
                pressure_range=(1e-6, 1e-3),  # ~10^-6 to 10^-3 Pa
                frequency_range=(1e6, 1e12),  # Extended frequency range
                strain_range=(-0.05, 0.05),
                humidity_range=(0, 0.1),  # Extremely dry
                magnetic_field=0.0  # No magnetic field in space
            ),
            'reentry_heating': CharacterizationCondition(
                name='Atmospheric Reentry Heating',
                temperature_range=(373, 2273),  # 100°C to 2000°C
                pressure_range=(101325, 1e6),  # Atmospheric reentry pressures
                frequency_range=(1e6, 1e11),
                strain_range=(-0.2, 0.2),
                humidity_range=(10, 50)
            ),
            'launch_vibration': CharacterizationCondition(
                name='Rocket Launch Vibration',
                temperature_range=(273, 293),  # 0-20°C
                pressure_range=(101325, 101325),
                frequency_range=(5, 10000),  # 5 Hz to 10 kHz vibration
                strain_range=(-0.5, 0.5),  # High vibration strains
                humidity_range=(40, 60)
            ),

            # Industrial and Harsh Environment Stress Factors
            'chemical_plant': CharacterizationCondition(
                name='Chemical Plant Environment',
                temperature_range=(293, 373),  # 20-100°C
                pressure_range=(101325, 5e5),  # Elevated pressures
                frequency_range=(1e6, 1e11),
                strain_range=(-0.05, 0.05),
                humidity_range=(60, 95)  # High humidity
            ),
            'offshore_oil_rig': CharacterizationCondition(
                name='Offshore Oil Rig Environment',
                temperature_range=(273, 313),  # 0-40°C
                pressure_range=(101325, 1e6),  # High pressures
                frequency_range=(1e6, 1e11),
                strain_range=(-0.1, 0.1),
                humidity_range=(70, 100)  # Marine environment
            ),
            'nuclear_reactor': CharacterizationCondition(
                name='Nuclear Reactor Environment',
                temperature_range=(293, 673),  # 20-400°C
                pressure_range=(1e7, 2e7),  # High pressures
                frequency_range=(1e6, 1e11),
                strain_range=(-0.02, 0.02),
                humidity_range=(10, 30),  # Low humidity
                magnetic_field=2.0,  # Moderate magnetic field
                electric_field=1e5  # High electric field
            ),

            # Emerging Technology Stress Factors
            'quantum_computing': CharacterizationCondition(
                name='Quantum Computing Environment',
                temperature_range=(0.01, 1),  # milliKelvin to 1K
                pressure_range=(1e-9, 1e-6),  # Ultra-high vacuum
                frequency_range=(1e9, 1e12),  # GHz to THz range
                strain_range=(-1e-6, 1e-6),  # Microstrains only
                humidity_range=(0, 0.001),  # Extremely dry
                magnetic_field=0.001,  # MicroTesla shielding
                electric_field=0.1  # Very low fields
            ),
            '5g_6g_environment': CharacterizationCondition(
                name='5G/6G Electromagnetic Environment',
                temperature_range=(253, 333),  # -20°C to 60°C
                pressure_range=(101325, 101325),
                frequency_range=(600e6, 300e9),  # 600 MHz to 300 GHz
                strain_range=(-0.01, 0.01),
                humidity_range=(20, 80),
                electric_field=1e5,  # 100 kV/m (high EMF)
                magnetic_field=0.01  # Low magnetic field
            ),
            'autonomous_vehicle': CharacterizationCondition(
                name='Autonomous Vehicle Environment',
                temperature_range=(233, 358),  # -40°C to 85°C
                pressure_range=(101325, 101325),
                frequency_range=(1e6, 1e11),
                strain_range=(-0.2, 0.2),  # Crash impact strains
                humidity_range=(10, 95),  # Full environmental range
                electric_field=1e4  # Moderate EMF from vehicle systems
            )
        }

    def create_digital_twin(self, metamaterial_design: Dict[str, Any]) -> DigitalTwin:
        """
        Create a digital twin from a metamaterial design

        Args:
            metamaterial_design: Dictionary containing metamaterial specification

        Returns:
            DigitalTwin object with virtual representation
        """

        twin_id = f"DT-{metamaterial_design['name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        digital_twin = DigitalTwin(
            name=f"Digital Twin: {metamaterial_design['name']}",
            original_design=metamaterial_design,
            twin_id=twin_id,
            creation_timestamp=datetime.now().isoformat(),
            characterization_results={},
            simulation_history=[],
            optimization_recommendations=[],
            failure_modes_identified=[],
            confidence_level=0.0,
            validation_status="created"
        )

        self.digital_twins[twin_id] = digital_twin

        logging.info(f"🤖 ECH0 created digital twin: {twin_id}")
        logging.info(f"   Based on design: {metamaterial_design['name']}")

        return digital_twin

    def characterize_digital_twin(self, digital_twin: DigitalTwin,
                                 conditions: List[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive characterization of the digital twin

        Args:
            digital_twin: The digital twin to characterize
            conditions: List of condition names to test under

        Returns:
            Comprehensive characterization results
        """

        if conditions is None:
            conditions = ['standard_lab']

        logging.info(f"\n🔬 ECH0 commencing characterization of {digital_twin.name}")
        logging.info(f"   Testing under conditions: {conditions}")

        results = {
            'twin_id': digital_twin.twin_id,
            'timestamp': datetime.now().isoformat(),
            'conditions_tested': conditions,
            'performance_metrics': {},
            'failure_analysis': {},
            'optimization_opportunities': {},
            'reliability_assessment': {},
            'cost_benefit_analysis': {}
        }

        # Perform characterization under each condition
        for condition_name in conditions:
            if condition_name in self.characterization_conditions:
                condition = self.characterization_conditions[condition_name]

                logging.info(f"   Testing under: {condition.name}")

                # Run multi-physics characterization
                condition_results = self._run_multiphysics_characterization(
                    digital_twin, condition
                )

                results['performance_metrics'][condition_name] = condition_results

                # Analyze failure modes
                failure_analysis = self._analyze_failure_modes(
                    digital_twin, condition, condition_results
                )
                results['failure_analysis'][condition_name] = failure_analysis

                # Identify optimization opportunities
                optimizations = self._identify_optimizations(
                    digital_twin, condition, condition_results
                )
                results['optimization_opportunities'][condition_name] = optimizations

        # Overall reliability assessment
        results['reliability_assessment'] = self._assess_overall_reliability(
            digital_twin, results
        )

        # Cost-benefit analysis
        results['cost_benefit_analysis'] = self._perform_cost_benefit_analysis(
            digital_twin, results
        )

        # Update digital twin with results
        digital_twin.characterization_results = results
        digital_twin.confidence_level = self._calculate_twin_confidence(results)
        digital_twin.validation_status = "characterized"
        digital_twin.optimization_recommendations = self._extract_recommendations(results)

        # Log simulation in history
        digital_twin.simulation_history.append({
            'timestamp': results['timestamp'],
            'type': 'comprehensive_characterization',
            'conditions': conditions,
            'key_findings': self._summarize_key_findings(results)
        })

        logging.info(f"✅ Characterization complete. Confidence level: {digital_twin.confidence_level:.2f}")

        return results

    def _run_multiphysics_characterization(self, digital_twin: DigitalTwin,
                                         condition: CharacterizationCondition) -> Dict[str, Any]:
        """Run multi-physics characterization simulation"""

        design = digital_twin.original_design
        category = design['category']

        results = {
            'temperature_effects': {},
            'pressure_effects': {},
            'performance_degradation': {},
            'operational_limits': {}
        }

        # Temperature characterization
        temp_range = np.linspace(condition.temperature_range[0],
                               condition.temperature_range[1], 10)

        for temp in temp_range:
            temp_results = self._simulate_temperature_effect(digital_twin, temp)
            results['temperature_effects'][f"{temp:.1f}K"] = temp_results

        # Pressure characterization
        pressure_range = np.logspace(np.log10(condition.pressure_range[0]),
                                   np.log10(condition.pressure_range[1]), 8)

        for pressure in pressure_range:
            pressure_results = self._simulate_pressure_effect(digital_twin, pressure)
            results['pressure_effects'][f"{pressure:.1e}Pa"] = pressure_results

        # Category-specific characterization
        if category == 'electromagnetic':
            results.update(self._characterize_electromagnetic(digital_twin, condition))
        elif category == 'acoustic':
            results.update(self._characterize_acoustic(digital_twin, condition))
        elif category == 'mechanical':
            results.update(self._characterize_mechanical(digital_twin, condition))

        # Performance degradation analysis
        results['performance_degradation'] = self._analyze_performance_degradation(
            digital_twin, condition
        )

        # Operational limits identification
        results['operational_limits'] = self._identify_operational_limits(
            digital_twin, condition, results
        )

        return results

    def _simulate_temperature_effect(self, digital_twin: DigitalTwin, temperature: float) -> Dict[str, Any]:
        """Simulate temperature effects on the metamaterial"""

        design = digital_twin.original_design
        results = {'thermal_expansion': 0.0, 'property_changes': {}, 'thermal_stability': True}

        # Calculate thermal expansion
        materials = design.get('material_composition', {})
        total_expansion = 0.0
        total_fraction = 0.0

        for material_name, fraction in materials.items():
            if material_name in self.material_models:
                # Simplified thermal expansion coefficient (typical values)
                expansion_coeff = 1.5e-5 if 'metal' in material_name.lower() else 5e-6  # 1/K
                total_expansion += expansion_coeff * fraction * (temperature - 293.15)
                total_fraction += fraction

                # Property temperature dependence
                if 'conductivity' in self.material_models[material_name]:
                    base_conductivity = self.material_models[material_name]['conductivity']
                    temp_coeff = self.material_models[material_name].get('temperature_coefficient', 0.004)
                    temp_conductivity = base_conductivity * (1 + temp_coeff * (temperature - 293.15))
                    results['property_changes'][f'{material_name}_conductivity'] = temp_conductivity

        results['thermal_expansion'] = total_expansion / total_fraction if total_fraction > 0 else 0.0

        # Thermal stability assessment
        if abs(results['thermal_expansion']) > 0.01:  # >1% expansion
            results['thermal_stability'] = False

        return results

    def _simulate_pressure_effect(self, digital_twin: DigitalTwin, pressure: float) -> Dict[str, Any]:
        """Simulate pressure effects on the metamaterial"""

        design = digital_twin.original_design
        results = {'compression': 0.0, 'bulk_modulus_effect': 0.0, 'pressure_stability': True}

        # Calculate compression effects
        materials = design.get('material_composition', {})
        total_compression = 0.0

        for material_name, fraction in materials.items():
            if material_name in self.material_models:
                # Simplified bulk modulus effects
                bulk_modulus = self.material_models[material_name].get('youngs_modulus', 70e9) / 3
                compression = (pressure / bulk_modulus) * 100  # percentage
                total_compression += compression * fraction

        results['compression'] = total_compression

        # Pressure stability assessment
        if results['compression'] > 5.0:  # >5% compression
            results['pressure_stability'] = False

        return results

    def _characterize_electromagnetic(self, digital_twin: DigitalTwin,
                                    condition: CharacterizationCondition) -> Dict[str, Any]:
        """Electromagnetic-specific characterization"""

        design = digital_twin.original_design
        unit_cell = design.get('unit_cell_design', {})

        results = {
            'frequency_response': {},
            'band_structure': {},
            'polarization_effects': {},
            'angle_dependence': {},
            'efficiency_metrics': {}
        }

        # Frequency sweep simulation
        if condition.frequency_range:
            freq_range = np.logspace(np.log10(condition.frequency_range[0]),
                                   np.log10(condition.frequency_range[1]), 50)

            transmission = []
            reflection = []

            for freq in freq_range:
                # Simplified frequency response model
                if unit_cell.get('template') == 'split_ring_resonator':
                    # Lorentzian resonance model
                    resonance_freq = unit_cell.get('dimensions', {}).get('resonance_frequency', 3.0) * 1e9
                    quality_factor = unit_cell.get('dimensions', {}).get('quality_factor', 30)

                    detuning = freq - resonance_freq
                    linewidth = resonance_freq / quality_factor

                    # Transmission magnitude (simplified)
                    t_mag = 1 / np.sqrt(1 + (2 * detuning / linewidth)**2)
                    r_mag = np.sqrt(1 - t_mag**2)

                elif unit_cell.get('template') == 'photonic_crystal':
                    # Photonic bandgap model
                    bandgap_center = unit_cell.get('dimensions', {}).get('bandgap_center', 500e12)
                    bandgap_width = unit_cell.get('dimensions', {}).get('bandgap_width', 100e12)

                    if abs(freq - bandgap_center) < bandgap_width / 2:
                        t_mag = 0.1  # Strong attenuation in bandgap
                        r_mag = 0.9
                    else:
                        t_mag = 0.95
                        r_mag = 0.05

                else:
                    # Generic response
                    t_mag = 0.8 + 0.2 * np.random.random()
                    r_mag = np.sqrt(1 - t_mag**2)

                transmission.append(float(t_mag))
                reflection.append(float(r_mag))

            results['frequency_response'] = {
                'frequencies': freq_range.tolist(),
                'transmission': transmission,
                'reflection': reflection
            }

        # Polarization and angle dependence
        angles = np.linspace(0, 90, 19)  # 0° to 90° in 5° steps
        polarizations = ['TE', 'TM']

        angle_response = {}
        for pol in polarizations:
            angle_response[pol] = [0.8 + 0.2 * np.sin(np.radians(angle)) for angle in angles]

        results['angle_dependence'] = {
            'angles': angles.tolist(),
            'polarization_response': angle_response
        }

        # Efficiency metrics
        results['efficiency_metrics'] = {
            'insertion_loss': np.mean(reflection) * 10,  # dB
            'bandwidth_efficiency': 0.75 + 0.25 * np.random.random(),
            'polarization_insensitive': 0.85 + 0.15 * np.random.random()
        }

        return results

    def _characterize_acoustic(self, digital_twin: DigitalTwin,
                             condition: CharacterizationCondition) -> Dict[str, Any]:
        """Acoustic-specific characterization"""

        design = digital_twin.original_design
        unit_cell = design.get('unit_cell_design', {})

        results = {
            'frequency_response': {},
            'sound_attenuation': {},
            'directionality': {},
            'impedance_matching': {},
            'nonlinearity': {}
        }

        # Acoustic frequency response
        if condition.frequency_range:
            freq_range = np.logspace(np.log10(max(condition.frequency_range[0], 20)),
                                   np.log10(min(condition.frequency_range[1], 20000)), 40)

            attenuation = []

            for freq in freq_range:
                if unit_cell.get('template') == 'sonic_crystal':
                    # Bragg scattering model
                    lattice_constant = unit_cell.get('dimensions', {}).get('lattice_constant', 0.1)
                    speed_of_sound = 343  # m/s in air

                    # Attenuation peaks at Bragg frequencies
                    bragg_freq = speed_of_sound / (2 * lattice_constant)
                    detuning = abs(freq - bragg_freq) / bragg_freq

                    if detuning < 0.1:  # Near Bragg frequency
                        atten = 30 + 20 * np.random.random()  # 30-50 dB attenuation
                    else:
                        atten = 5 + 10 * np.random.random()  # 5-15 dB

                elif unit_cell.get('template') == 'membrane_resonator':
                    # Membrane resonance model
                    resonance_freq = unit_cell.get('dimensions', {}).get('resonance_frequency', 1000)

                    # Quality factor determines bandwidth
                    q_factor = unit_cell.get('dimensions', {}).get('quality_factor', 10)
                    detuning = abs(freq - resonance_freq) / (resonance_freq / q_factor)

                    if detuning < 1:
                        atten = 25 / (1 + detuning**2)  # Lorentzian response
                    else:
                        atten = 2 + 3 * np.random.random()

                else:
                    atten = 5 + 15 * np.random.random()

                attenuation.append(float(atten))

            results['frequency_response'] = {
                'frequencies': freq_range.tolist(),
                'attenuation_db': attenuation
            }

        # Sound attenuation metrics
        results['sound_attenuation'] = {
            'average_attenuation': np.mean(attenuation) if 'attenuation' in locals() else 10.0,
            'peak_attenuation': np.max(attenuation) if 'attenuation' in locals() else 30.0,
            'effective_bandwidth': 0.6 + 0.4 * np.random.random()
        }

        # Impedance matching
        materials = design.get('material_composition', {})
        air_impedance = 415  # kg/(m²·s) for air

        material_impedance = 0
        for material_name, fraction in materials.items():
            if material_name in self.material_models:
                mat_impedance = self.material_models[material_name].get('acoustic_impedance', air_impedance)
                material_impedance += mat_impedance * fraction

        impedance_ratio = material_impedance / air_impedance
        results['impedance_matching'] = {
            'material_impedance': material_impedance,
            'impedance_ratio': impedance_ratio,
            'reflection_coefficient': abs(impedance_ratio - 1) / (impedance_ratio + 1)
        }

        return results

    def _characterize_mechanical(self, digital_twin: DigitalTwin,
                               condition: CharacterizationCondition) -> Dict[str, Any]:
        """Mechanical-specific characterization"""

        design = digital_twin.original_design
        unit_cell = design.get('unit_cell_design', {})

        results = {
            'stress_strain_response': {},
            'energy_absorption': {},
            'poisson_ratio_analysis': {},
            'fatigue_behavior': {},
            'impact_resistance': {}
        }

        # Stress-strain characterization
        if condition.strain_range:
            strain_range = np.linspace(condition.strain_range[0],
                                     condition.strain_range[1], 50)

            stress_response = []
            for strain in strain_range:
                if unit_cell.get('template') == 'auxetic_structure':
                    # Negative Poisson's ratio behavior
                    poissons_ratio = -0.3  # Negative for auxetic
                    youngs_modulus = 50e6  # Pa, typical for auxetic materials

                    # Non-linear stress-strain with auxetic effects
                    stress = youngs_modulus * strain * (1 + 0.5 * abs(strain))

                elif unit_cell.get('template') == 'pentamode_material':
                    # Near-liquid behavior
                    poissons_ratio = -0.1
                    youngs_modulus = 10e6  # Very low for pentamode

                    # Highly compliant response
                    stress = youngs_modulus * strain

                else:
                    # Generic mechanical response
                    poissons_ratio = 0.3
                    youngs_modulus = 100e6
                    stress = youngs_modulus * strain * (1 + 0.1 * strain**2)

                stress_response.append(float(stress))

            results['stress_strain_response'] = {
                'strain': strain_range.tolist(),
                'stress': stress_response,
                'youngs_modulus': youngs_modulus,
                'poissons_ratio': poissons_ratio
            }

        # Energy absorption
        if 'stress' in results.get('stress_strain_response', {}):
            stress = results['stress_strain_response']['stress']
            strain = results['stress_strain_response']['strain']

            # Calculate energy absorption (area under stress-strain curve)
            energy_absorption = np.trapz(stress, strain)
            max_energy = max(stress) * max(strain)

            results['energy_absorption'] = {
                'total_energy_absorbed': energy_absorption,
                'energy_absorption_efficiency': energy_absorption / max_energy if max_energy > 0 else 0,
                'specific_energy_absorption': energy_absorption / 1000  # J/kg, approximate
            }

        # Fatigue behavior
        results['fatigue_behavior'] = {
            'fatigue_life': 10000 + 50000 * np.random.random(),  # cycles
            'fatigue_strength': 0.5 + 0.3 * np.random.random(),  # fraction of ultimate strength
            'crack_growth_rate': 1e-9 + 1e-8 * np.random.random()  # m/cycle
        }

        # Impact resistance
        results['impact_resistance'] = {
            'impact_energy_absorption': 50 + 50 * np.random.random(),  # J
            'peak_force_attenuation': 0.7 + 0.3 * np.random.random(),  # fraction
            'deformation_recovery': 0.8 + 0.2 * np.random.random()  # fraction
        }

        return results

    def _analyze_performance_degradation(self, digital_twin: DigitalTwin,
                                       condition: CharacterizationCondition) -> Dict[str, Any]:
        """Analyze performance degradation over time/conditions"""

        design = digital_twin.original_design
        category = design['category']

        degradation = {
            'degradation_rate': 0.0,
            'lifetime_prediction': 0.0,
            'failure_probability': 0.0,
            'maintenance_intervals': 0.0
        }

        # Category-specific degradation models
        if category == 'electromagnetic':
            # EM degradation due to oxidation, corrosion
            degradation['degradation_rate'] = 0.001 + 0.005 * np.random.random()  # %/day
            degradation['lifetime_prediction'] = 5 + 10 * np.random.random()  # years
            degradation['failure_probability'] = 0.05 + 0.15 * np.random.random()

        elif category == 'acoustic':
            # Acoustic degradation due to fatigue, delamination
            degradation['degradation_rate'] = 0.0005 + 0.002 * np.random.random()  # %/day
            degradation['lifetime_prediction'] = 8 + 12 * np.random.random()  # years
            degradation['failure_probability'] = 0.03 + 0.12 * np.random.random()

        elif category == 'mechanical':
            # Mechanical degradation due to fatigue, creep
            degradation['degradation_rate'] = 0.002 + 0.008 * np.random.random()  # %/day
            degradation['lifetime_prediction'] = 3 + 7 * np.random.random()  # years
            degradation['failure_probability'] = 0.08 + 0.17 * np.random.random()

        degradation['maintenance_intervals'] = degradation['lifetime_prediction'] / 10  # years

        return degradation

    def _identify_operational_limits(self, digital_twin: DigitalTwin,
                                   condition: CharacterizationCondition,
                                   characterization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identify operational limits and boundaries"""

        limits = {
            'temperature_limits': condition.temperature_range,
            'pressure_limits': condition.pressure_range,
            'frequency_limits': condition.frequency_range,
            'strain_limits': condition.strain_range,
            'critical_failure_points': [],
            'safe_operating_envelope': {}
        }

        # Analyze characterization results for critical points
        temp_effects = characterization_results.get('temperature_effects', {})
        for temp_key, temp_result in temp_effects.items():
            if not temp_result.get('thermal_stability', True):
                limits['critical_failure_points'].append({
                    'type': 'thermal_instability',
                    'condition': temp_key,
                    'description': f'Thermal expansion exceeds stability threshold at {temp_key}'
                })

        pressure_effects = characterization_results.get('pressure_effects', {})
        for pressure_key, pressure_result in pressure_effects.items():
            if not pressure_result.get('pressure_stability', True):
                limits['critical_failure_points'].append({
                    'type': 'pressure_instability',
                    'condition': pressure_key,
                    'description': f'Compression exceeds stability threshold at {pressure_key}'
                })

        # Define safe operating envelope
        limits['safe_operating_envelope'] = {
            'temperature': (condition.temperature_range[0] * 0.9, condition.temperature_range[1] * 0.9),
            'pressure': (condition.pressure_range[0] * 0.8, condition.pressure_range[1] * 0.8),
            'recommended_monitoring_frequency': 'monthly',
            'emergency_shutdown_triggers': ['temperature > 350K', 'pressure > 1e7 Pa']
        }

        return limits

    def _analyze_failure_modes(self, digital_twin: DigitalTwin,
                             condition: CharacterizationCondition,
                             characterization_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze potential failure modes"""

        failure_modes = []

        # Thermal failure modes
        temp_range = condition.temperature_range
        if temp_range[1] - temp_range[0] > 100:  # Large temperature variation
            failure_modes.append({
                'mode': 'thermal_cycling_fatigue',
                'severity': 'medium',
                'probability': 0.3,
                'description': 'Repeated thermal expansion/contraction may cause material fatigue',
                'mitigation': 'Implement thermal management system'
            })

        # Pressure failure modes
        pressure_range = condition.pressure_range
        if pressure_range[1] > 1e6:  # High pressure
            failure_modes.append({
                'mode': 'pressure_induced_deformation',
                'severity': 'high',
                'probability': 0.6,
                'description': 'High pressure may cause permanent deformation of unit cells',
                'mitigation': 'Reinforce unit cell structure or reduce operational pressure'
            })

        # Material-specific failure modes
        materials = digital_twin.original_design.get('material_composition', {})
        if 'copper' in materials:
            failure_modes.append({
                'mode': 'oxidation_corrosion',
                'severity': 'medium',
                'probability': 0.4,
                'description': 'Copper oxidation may degrade electromagnetic performance',
                'mitigation': 'Apply protective coating or operate in inert atmosphere'
            })

        # Category-specific failure modes
        category = digital_twin.original_design['category']
        if category == 'electromagnetic':
            failure_modes.append({
                'mode': 'resonance_drift',
                'severity': 'low',
                'probability': 0.2,
                'description': 'Environmental conditions may shift resonance frequencies',
                'mitigation': 'Implement active frequency tuning system'
            })
        elif category == 'acoustic':
            failure_modes.append({
                'mode': 'acoustic_fatigue',
                'severity': 'medium',
                'probability': 0.35,
                'description': 'High-intensity acoustic waves may cause material fatigue',
                'mitigation': 'Reduce acoustic power or implement periodic inspection'
            })
        elif category == 'mechanical':
            failure_modes.append({
                'mode': 'impact_damage',
                'severity': 'high',
                'probability': 0.5,
                'description': 'High strain rates may cause sudden failure',
                'mitigation': 'Implement impact protection or limit strain rates'
            })

        return failure_modes

    def _identify_optimizations(self, digital_twin: DigitalTwin,
                              condition: CharacterizationCondition,
                              characterization_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities"""

        optimizations = []

        # Performance-based optimizations
        perf_metrics = characterization_results.get('performance_metrics', {})

        # Frequency response optimization
        if 'frequency_response' in perf_metrics.get('standard_lab', {}):
            freq_response = perf_metrics['standard_lab']['frequency_response']
            transmission = freq_response.get('transmission', [])
            if transmission:
                avg_transmission = np.mean(transmission)
                if avg_transmission < 0.7:
                    optimizations.append({
                        'type': 'transmission_optimization',
                        'potential_improvement': f"{(0.85 - avg_transmission) * 100:.1f}% increase",
                        'method': 'Adjust unit cell geometry parameters',
                        'expected_benefit': 'Improved signal transmission efficiency'
                    })

        # Material composition optimization
        materials = digital_twin.original_design.get('material_composition', {})
        if len(materials) > 3:
            optimizations.append({
                'type': 'material_simplification',
                'potential_improvement': '20-30% cost reduction',
                'method': 'Reduce number of material types while maintaining performance',
                'expected_benefit': 'Lower manufacturing complexity and cost'
            })

        # Thermal management optimization
        temp_effects = characterization_results.get('temperature_effects', {})
        if any(not effect.get('thermal_stability', True) for effect in temp_effects.values()):
            optimizations.append({
                'type': 'thermal_stability_improvement',
                'potential_improvement': 'Extended operational temperature range',
                'method': 'Add thermal expansion compensation or change material composition',
                'expected_benefit': 'Improved reliability in variable temperature environments'
            })

        # Structural optimization
        if digital_twin.original_design['category'] == 'mechanical':
            optimizations.append({
                'type': 'topology_optimization',
                'potential_improvement': '15-25% weight reduction',
                'method': 'Optimize unit cell topology for strength-to-weight ratio',
                'expected_benefit': 'Lighter, more efficient mechanical metamaterials'
            })

        return optimizations

    def _assess_overall_reliability(self, digital_twin: DigitalTwin,
                                  characterization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall reliability of the digital twin/metamaterial"""

        reliability = {
            'overall_score': 0.0,
            'confidence_intervals': {},
            'reliability_factors': {},
            'recommended_testing': []
        }

        # Calculate reliability score based on various factors
        factors = {
            'material_stability': 0.8,
            'performance_consistency': 0.7,
            'failure_mode_coverage': 0.6,
            'environmental_robustness': 0.75
        }

        # Adjust factors based on characterization results
        failure_modes = characterization_results.get('failure_analysis', {})
        if failure_modes:
            high_severity_failures = sum(1 for condition in failure_modes.values()
                                       for failure in condition
                                       if failure.get('severity') == 'high')
            factors['failure_mode_coverage'] = max(0.3, 1.0 - high_severity_failures * 0.1)

        # Environmental robustness
        conditions_tested = len(characterization_results.get('conditions_tested', []))
        factors['environmental_robustness'] = min(1.0, 0.5 + conditions_tested * 0.1)

        # Calculate overall score
        reliability['overall_score'] = np.mean(list(factors.values()))
        reliability['reliability_factors'] = factors

        # Confidence intervals
        reliability['confidence_intervals'] = {
            'lower_bound': reliability['overall_score'] * 0.8,
            'upper_bound': min(1.0, reliability['overall_score'] * 1.2)
        }

        # Recommended testing
        reliability['recommended_testing'] = [
            'Accelerated life testing under extreme conditions',
            'Cyclic loading tests for fatigue analysis',
            'Environmental chamber testing for temperature/pressure extremes',
            'Non-destructive evaluation for defect detection'
        ]

        return reliability

    def _perform_cost_benefit_analysis(self, digital_twin: DigitalTwin,
                                     characterization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cost-benefit analysis"""

        design = digital_twin.original_design

        analysis = {
            'manufacturing_cost_estimate': 0.0,
            'performance_value': 0.0,
            'roi_projection': 0.0,
            'break_even_analysis': {},
            'scalability_assessment': {}
        }

        # Manufacturing cost estimate
        materials = design.get('material_composition', {})
        fabrication_method = design.get('fabrication_method', 'generic')

        base_cost_per_kg = {
            'copper': 8.0,  # $/kg
            'aluminum': 2.5,
            'silicon_dioxide': 1.0,
            'polystyrene': 3.0
        }

        total_material_cost = 0
        for material, fraction in materials.items():
            cost_per_kg = base_cost_per_kg.get(material, 5.0)
            total_material_cost += cost_per_kg * fraction

        # Fabrication cost multiplier
        fabrication_multipliers = {
            'photolithography_with_metal_deposition': 50.0,
            '3D_printing_with_conductive_ink': 20.0,
            'stereolithography_with_acoustic_tuning': 15.0,
            'advanced_additive_manufacturing': 25.0
        }

        fab_multiplier = fabrication_multipliers.get(fabrication_method, 10.0)
        analysis['manufacturing_cost_estimate'] = total_material_cost * fab_multiplier

        # Performance value assessment
        category = design['category']
        if category == 'electromagnetic':
            analysis['performance_value'] = 10000 + 50000 * np.random.random()  # High value for EM applications
        elif category == 'acoustic':
            analysis['performance_value'] = 5000 + 15000 * np.random.random()
        elif category == 'mechanical':
            analysis['performance_value'] = 8000 + 20000 * np.random.random()

        # ROI projection
        annual_benefit = analysis['performance_value'] * 0.3  # 30% of performance value as benefit
        annual_cost = analysis['manufacturing_cost_estimate'] * 0.1  # 10% of initial cost as maintenance
        analysis['roi_projection'] = (annual_benefit - annual_cost) / analysis['manufacturing_cost_estimate']

        # Break-even analysis
        analysis['break_even_analysis'] = {
            'initial_investment': analysis['manufacturing_cost_estimate'],
            'annual_operating_cost': annual_cost,
            'annual_benefit': annual_benefit,
            'break_even_period_years': analysis['manufacturing_cost_estimate'] / annual_benefit if annual_benefit > 0 else float('inf')
        }

        # Scalability assessment
        analysis['scalability_assessment'] = {
            'production_volume_potential': '1000-10000 units/year',
            'cost_reduction_with_scale': '30-50% reduction at 1000 units',
            'technology_readiness': 'TRL 4-6 (lab validation to prototype)',
            'market_adoption_barrier': 'medium' if analysis['roi_projection'] > 0.5 else 'high'
        }

        return analysis

    def _calculate_twin_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence level in the digital twin"""

        confidence = 0.5  # Base confidence

        # Factors increasing confidence
        conditions_tested = len(results.get('conditions_tested', []))
        confidence += min(0.2, conditions_tested * 0.05)

        # Performance metrics availability
        perf_metrics = results.get('performance_metrics', {})
        if perf_metrics:
            confidence += 0.15

        # Failure analysis completeness
        failure_analysis = results.get('failure_analysis', {})
        if failure_analysis:
            confidence += 0.1

        # Reliability assessment
        reliability = results.get('reliability_assessment', {})
        if reliability:
            reliability_score = reliability.get('overall_score', 0.5)
            confidence += reliability_score * 0.1

        return min(confidence, 1.0)

    def _extract_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Extract key recommendations from characterization results"""

        recommendations = []

        # From optimization opportunities
        optimizations = results.get('optimization_opportunities', {})
        for condition, opts in optimizations.items():
            for opt in opts:
                recommendations.append(f"{opt['type']}: {opt['method']} - {opt['expected_benefit']}")

        # From failure modes
        failure_analysis = results.get('failure_analysis', {})
        for condition, failures in failure_analysis.items():
            for failure in failures:
                if failure['severity'] in ['high', 'medium']:
                    recommendations.append(f"Address {failure['mode']}: {failure['mitigation']}")

        # From cost-benefit analysis
        cost_benefit = results.get('cost_benefit_analysis', {})
        roi = cost_benefit.get('roi_projection', 0)
        if roi > 1.0:
            recommendations.append("High ROI opportunity - proceed with prototyping")
        elif roi < 0.3:
            recommendations.append("Low ROI - consider design modifications before prototyping")

        return recommendations[:10]  # Limit to top 10 recommendations

    def _summarize_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Summarize key findings from characterization"""

        findings = []

        # Performance highlights
        perf_metrics = results.get('performance_metrics', {})
        if perf_metrics:
            findings.append("Performance metrics characterized across multiple conditions")

        # Reliability assessment
        reliability = results.get('reliability_assessment', {})
        if reliability:
            score = reliability.get('overall_score', 0)
            findings.append(f"Reliability score: {score:.2f}/1.0")

        # Cost-benefit insights
        cost_benefit = results.get('cost_benefit_analysis', {})
        if cost_benefit:
            roi = cost_benefit.get('roi_projection', 0)
            findings.append(f"ROI projection: {roi:.2f}")

        # Critical issues
        failure_analysis = results.get('failure_analysis', {})
        critical_count = 0
        for condition, failures in failure_analysis.items():
            critical_count += sum(1 for f in failures if f.get('severity') == 'high')

        if critical_count > 0:
            findings.append(f"Identified {critical_count} critical failure modes requiring attention")

        return findings

    def run_digital_twin_campaign(self, metamaterial_designs: List[Dict[str, Any]],
                                conditions: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive digital twin characterization campaign"""

        if conditions is None:
            # Use comprehensive set of stress factors covering all major categories
            conditions = [
                'standard_lab',  # Baseline
                'extreme_temperature', 'cryogenic_temperature', 'high_temperature',  # Temperature stress
                'high_pressure', 'ultra_high_pressure', 'vacuum_conditions',  # Pressure stress
                'saltwater_corrosion', 'acid_exposure', 'alkaline_exposure', 'oxidizing_atmosphere',  # Chemical stress
                'vibration_testing', 'shock_testing', 'fatigue_cycling',  # Mechanical stress
                'gamma_radiation', 'neutron_radiation', 'uv_exposure', 'electromagnetic_interference',  # Radiation/EM stress
                'biodegradation', 'humidity_cycling', 'thermal_shock',  # Environmental stress
                'space_vacuum', 'reentry_heating', 'launch_vibration',  # Space/Aerospace stress
                'chemical_plant', 'offshore_oil_rig', 'nuclear_reactor',  # Industrial stress
                'quantum_computing', '5g_6g_environment', 'autonomous_vehicle'  # Emerging tech stress
            ]

        logging.info("🤖 ECH0 DIGITAL TWIN CHARACTERIZATION CAMPAIGN")
        logging.info("=" * 60)
        logging.info(f"Characterizing {len(metamaterial_designs)} metamaterials")
        logging.info(f"Test conditions: {conditions}")
        logging.info()

        campaign_results = {
            'timestamp': datetime.now().isoformat(),
            'designs_characterized': len(metamaterial_designs),
            'conditions_tested': conditions,
            'digital_twins_created': [],
            'campaign_summary': {},
            'top_performers': [],
            'critical_findings': []
        }

        for i, design in enumerate(metamaterial_designs):
            logging.info(f"\n🎯 CHARACTERIZING DESIGN {i+1}/{len(metamaterial_designs)}")
            logging.info(f"   {design['name']} ({design['category']})")

            # Create digital twin
            digital_twin = self.create_digital_twin(design)

            # Run characterization
            characterization_results = self.characterize_digital_twin(
                digital_twin, conditions
            )

            # Store results
            twin_summary = {
                'twin_id': digital_twin.twin_id,
                'original_design': design['name'],
                'category': design['category'],
                'confidence_level': digital_twin.confidence_level,
                'validation_status': digital_twin.validation_status,
                'key_metrics': self._extract_key_metrics(characterization_results),
                'critical_issues': self._extract_critical_issues(characterization_results),
                'recommendations': digital_twin.optimization_recommendations[:5]
            }

            campaign_results['digital_twins_created'].append(twin_summary)

        # Generate campaign summary
        campaign_results['campaign_summary'] = self._generate_campaign_summary(
            campaign_results['digital_twins_created']
        )

        # Identify top performers
        campaign_results['top_performers'] = sorted(
            campaign_results['digital_twins_created'],
            key=lambda x: x['confidence_level'],
            reverse=True
        )[:5]

        # Extract critical findings
        campaign_results['critical_findings'] = self._extract_campaign_critical_findings(
            campaign_results['digital_twins_created']
        )

        logging.info(f"\n🏆 CAMPAIGN COMPLETE")
        logging.info(f"Digital twins created: {len(campaign_results['digital_twins_created'])}")
        logging.info(f"Average confidence: {campaign_results['campaign_summary']['average_confidence']:.2f}")
        logging.info(f"Top performer: {campaign_results['top_performers'][0]['original_design']}")

        return campaign_results

    def _extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from characterization results"""

        metrics = {}

        # Performance metrics
        perf = results.get('performance_metrics', {}).get('standard_lab', {})
        if 'efficiency_metrics' in perf:
            metrics.update(perf['efficiency_metrics'])

        # Reliability
        reliability = results.get('reliability_assessment', {})
        if reliability:
            metrics['reliability_score'] = reliability.get('overall_score', 0)

        # Cost-benefit
        cost_benefit = results.get('cost_benefit_analysis', {})
        if cost_benefit:
            metrics['roi_projection'] = cost_benefit.get('roi_projection', 0)
            metrics['manufacturing_cost'] = cost_benefit.get('manufacturing_cost_estimate', 0)

        return metrics

    def _extract_critical_issues(self, results: Dict[str, Any]) -> List[str]:
        """Extract critical issues from characterization"""

        issues = []

        # Failure modes
        failure_analysis = results.get('failure_analysis', {})
        for condition, failures in failure_analysis.items():
            for failure in failures:
                if failure.get('severity') in ['high', 'medium']:
                    issues.append(f"{failure['mode']} ({condition}): {failure['description']}")

        # Operational limits
        for condition, perf in results.get('performance_metrics', {}).items():
            operational_limits = perf.get('operational_limits', {})
            critical_points = operational_limits.get('critical_failure_points', [])
            issues.extend([f"{point['type']} ({condition}): {point['description']}" for point in critical_points])

        return issues[:5]  # Top 5 critical issues

    def _generate_campaign_summary(self, twin_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall campaign summary"""

        summary = {
            'total_twins': len(twin_summaries),
            'average_confidence': np.mean([t['confidence_level'] for t in twin_summaries]),
            'categories_covered': list(set(t['category'] for t in twin_summaries)),
            'validation_status_distribution': {},
            'top_category': '',
            'critical_issues_count': 0
        }

        # Validation status distribution
        statuses = {}
        for twin in twin_summaries:
            status = twin['validation_status']
            statuses[status] = statuses.get(status, 0) + 1
        summary['validation_status_distribution'] = statuses

        # Top performing category
        category_performance = {}
        for twin in twin_summaries:
            cat = twin['category']
            category_performance[cat] = category_performance.get(cat, [])
            category_performance[cat].append(twin['confidence_level'])

        avg_by_category = {cat: np.mean(scores) for cat, scores in category_performance.items()}
        summary['top_category'] = max(avg_by_category, key=avg_by_category.get)

        # Critical issues count
        summary['critical_issues_count'] = sum(len(t.get('critical_issues', [])) for t in twin_summaries)

        return summary

    def _extract_campaign_critical_findings(self, twin_summaries: List[Dict[str, Any]]) -> List[str]:
        """Extract critical findings across the entire campaign"""

        findings = []

        # Most common issues
        all_issues = []
        for twin in twin_summaries:
            all_issues.extend(twin.get('critical_issues', []))

        if all_issues:
            # Count frequency of each issue type
            issue_counts = {}
            for issue in all_issues:
                issue_type = issue.split(':')[0] if ':' in issue else issue
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

            # Top 3 most common issues
            top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            findings.extend([f"Most common issue: {issue} ({count} occurrences)" for issue, count in top_issues])

        # Best and worst performers
        confidences = [t['confidence_level'] for t in twin_summaries]
        findings.append(f"Confidence range: {min(confidences):.2f} - {max(confidences):.2f}")

        # Category insights
        category_counts = {}
        for twin in twin_summaries:
            cat = twin['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1

        findings.append(f"Category distribution: {category_counts}")

        return findings

    def export_digital_twin_results(self, campaign_results: Dict[str, Any], filename: str):
        """Export digital twin characterization results"""

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'campaign_results': campaign_results,
            'digital_twin_details': {}
        }

        # Add detailed twin information
        for twin_summary in campaign_results['digital_twins_created']:
            twin_id = twin_summary['twin_id']
            if twin_id in self.digital_twins:
                export_data['digital_twin_details'][twin_id] = {
                    'twin_object': self.digital_twins[twin_id],
                    'full_characterization': self.digital_twins[twin_id].characterization_results
                }

        # Save to file (excluding non-serializable objects)
        serializable_data = {
            'export_timestamp': export_data['export_timestamp'],
            'campaign_results': campaign_results,
            'summary_statistics': {
                'total_twins': len(campaign_results['digital_twins_created']),
                'average_confidence': campaign_results['campaign_summary']['average_confidence'],
                'top_performers': [t['original_design'] for t in campaign_results['top_performers'][:3]]
            }
        }

        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)

        logging.info(f"✅ Exported digital twin results to {filename}")


def main():
    """Run ECH0 digital twin characterization campaign"""

    logging.info("🤖 ECH0 DIGITAL TWIN CHARACTERIZER")
    logging.info("=" * 50)

    characterizer = ECH0_DigitalTwinCharacterizer()

    # Load metamaterial designs from previous campaign
    try:
        with open('ech0_metamaterials_results.json', 'r') as f:
            previous_results = json.load(f)
            metamaterial_designs = previous_results.get('designs', [])

            if not metamaterial_designs:
                logging.info("No metamaterial designs found. Generating sample designs...")
                # Generate sample designs for demonstration
                metamaterial_designs = [
                    {
                        'name': 'ECH0-ELE-HYBRID-2',
                        'category': 'electromagnetic',
                        'unit_cell_design': {
                            'template': 'split_ring_resonator',
                            'dimensions': {'resonance_frequency': 3.2, 'quality_factor': 35}
                        },
                        'material_composition': {'copper': 0.7, 'silicon_dioxide': 0.2, 'air': 0.1},
                        'fabrication_method': 'photolithography_with_metal_deposition'
                    },
                    {
                        'name': 'ECH0-ACO-NOVEL-HIERARCHICAL-2',
                        'category': 'acoustic',
                        'unit_cell_design': {
                            'template': 'sonic_crystal',
                            'dimensions': {'lattice_constant': 0.1}
                        },
                        'material_composition': {'polystyrene': 0.8, 'air': 0.2},
                        'fabrication_method': 'stereolithography_with_acoustic_tuning'
                    },
                    {
                        'name': 'ECH0-MEC-PENTAMODE-MATERIAL-2',
                        'category': 'mechanical',
                        'unit_cell_design': {
                            'template': 'pentamode_material',
                            'dimensions': {'bulk_modulus': 0.1e9}
                        },
                        'material_composition': {'aluminum': 0.7, 'rubber': 0.3},
                        'fabrication_method': 'advanced_additive_manufacturing'
                    }
                ]
    except FileNotFoundError:
        logging.info("Previous results not found. Using sample metamaterial designs...")
        metamaterial_designs = [
            {
                'name': 'Sample-Electromagnetic-Metamaterial',
                'category': 'electromagnetic',
                'unit_cell_design': {'template': 'split_ring_resonator'},
                'material_composition': {'copper': 0.6, 'silicon_dioxide': 0.4},
                'fabrication_method': 'photolithography_with_metal_deposition'
            },
            {
                'name': 'Sample-Acoustic-Metamaterial',
                'category': 'acoustic',
                'unit_cell_design': {'template': 'sonic_crystal'},
                'material_composition': {'polystyrene': 0.8, 'air': 0.2},
                'fabrication_method': 'stereolithography_with_acoustic_tuning'
            },
            {
                'name': 'Sample-Mechanical-Metamaterial',
                'category': 'mechanical',
                'unit_cell_design': {'template': 'auxetic_structure'},
                'material_composition': {'aluminum': 0.7, 'rubber': 0.3},
                'fabrication_method': 'advanced_additive_manufacturing'
            }
        ]

    # Run comprehensive digital twin characterization campaign with extensive stress factors
    campaign_results = characterizer.run_digital_twin_campaign(
        metamaterial_designs=metamaterial_designs,
        conditions=None  # Use all 27 comprehensive stress factor conditions
    )

    # Export results
    characterizer.export_digital_twin_results(
        campaign_results, 'ech0_digital_twin_results.json'
    )

    logging.info("\n🎊 ECH0 Digital Twin Characterization Campaign Complete!")
    logging.info("Results saved to ech0_digital_twin_results.json")


if __name__ == "__main__":
    main()
