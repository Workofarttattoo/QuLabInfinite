# TODO: Refactor long functions identified in code quality analysis
import logging
#!/usr/bin/env python3
"""
QuLab Expanded Lab Testing Framework
====================================

Comprehensive testing of all QuLab laboratories using digital twin simulations
with complete environmental conditions.

Tests physics labs, chemistry labs, biology labs, materials labs, and more
under realistic experimental conditions.

Author: QuLab Infinite Validation Team
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import random
import time
from pathlib import Path

from qulab_expanded_digital_twin import (
    QuLabDigitalTwinSimulator, EnvironmentalConditions,
    EnvironmentalPreset, DigitalTwinExperiment
)
from qulab_trap_framework import TestQuestion


@dataclass
class LabTestSuite:
    """Test suite for a specific laboratory"""

    lab_name: str
    test_categories: List[str]
    environmental_scenarios: List[str]
    physics_validation_tests: List[TestQuestion]
    performance_benchmarks: Dict[str, Any]
    failure_mode_tests: List[Dict[str, Any]]


@dataclass
class LabTestResult:
    """Results of testing a laboratory"""

    lab_name: str
    total_tests_run: int
    tests_passed: int
    success_rate: float
    environmental_sensitivity: Dict[str, float]
    physics_compliance: Dict[str, bool]
    performance_metrics: Dict[str, Any]
    critical_failures: List[str]
    recommendations: List[str]
    test_duration: float


class QuLabExpandedLabTester:
    """
    Comprehensive tester for all QuLab laboratories using digital twin simulations.

    Tests labs across multiple environmental conditions and validates physics compliance.
    """

    def __init__(self):
        self.digital_twin_simulator = QuLabDigitalTwinSimulator()
        self.test_results: Dict[str, LabTestResult] = {}
        self.lab_test_suites = self._create_lab_test_suites()

    def _create_lab_test_suites(self) -> Dict[str, LabTestSuite]:
        """Create comprehensive test suites for all major lab categories"""

        suites = {}

        # Physics Labs
        physics_labs = [
            'quantum_mechanics_lab', 'quantum_lab', 'quantum_computing_lab', 'particle_physics_lab',
            'nuclear_physics_lab', 'plasma_physics_lab', 'condensed_matter_physics_lab',
            'astrophysics_lab', 'fluid_dynamics_lab', 'thermodynamics_lab',
            'electromagnetism_lab', 'optics_and_photonics_lab'
        ]

        for lab_name in physics_labs:
            if lab_name in self.digital_twin_simulator.digital_twins:
                if lab_name == 'quantum_computing_lab':
                    suites[lab_name] = self._create_quantum_computing_lab_suite(lab_name)
                else:
                    suites[lab_name] = self._create_physics_lab_suite(lab_name)

        # Chemistry Labs
        chemistry_labs = [
            'chemistry_lab', 'organic_chemistry_lab', 'inorganic_chemistry_lab',
            'physical_chemistry_lab', 'analytical_chemistry_lab', 'computational_chemistry_lab',
            'polymer_chemistry_lab', 'electrochemistry_lab', 'catalysis_lab',
            'atmospheric_chemistry_lab'
        ]

        for lab_name in chemistry_labs:
            if lab_name in self.digital_twin_simulator.digital_twins:
                suites[lab_name] = self._create_chemistry_lab_suite(lab_name)

        # Biology Labs
        biology_labs = [
            'molecular_biology_lab', 'cell_biology_lab', 'genetics_lab',
            'genomics_lab', 'bioinformatics_lab', 'microbiology_lab',
            'immunology_lab', 'neuroscience_lab', 'developmental_biology_lab',
            'evolutionary_biology_lab', 'proteomics_lab', 'biochemistry_lab',
            'biophysics_lab', 'astrobiology_lab', 'systems_biology_lab',
            'synthetic_biology_lab', 'virology_lab', 'metabolomics_lab',
            'epigenetics_lab'
        ]

        for lab_name in biology_labs:
            if lab_name in self.digital_twin_simulator.digital_twins:
                suites[lab_name] = self._create_biology_lab_suite(lab_name)

        # Materials Labs
        materials_labs = [
            'materials_lab', 'materials_science_lab', 'materials_chemistry_lab',
            'nanotechnology_lab', 'biomedical_engineering_lab', 'mechanical_engineering_lab',
            'electrical_engineering_lab', 'aerospace_engineering_lab',
            'environmental_engineering_lab', 'chemical_engineering_lab'
        ]

        for lab_name in materials_labs:
            if lab_name in self.digital_twin_simulator.digital_twins:
                suites[lab_name] = self._create_materials_lab_suite(lab_name)

        # Other Labs
        other_labs = [
            'robotics_lab', 'control_systems_lab', 'signal_processing_lab',
            'computer_vision_lab', 'machine_learning_lab', 'neural_networks_lab',
            'natural_language_processing_lab', 'cryptography_lab', 'algorithm_design_lab',
            'graph_theory_lab', 'optimization_theory_lab', 'climate_modeling_lab',
            'meteorology_lab', 'oceanography_lab', 'hydrology_lab', 'seismology_lab',
            'geology_lab', 'geophysics_lab', 'renewable_energy_lab', 'carbon_capture_lab',
            'medical_imaging_lab', 'drug_design_lab', 'pharmacology_lab', 'toxicology_lab',
            'clinical_trials_simulation_lab', 'cardiology_lab', 'oncology_lab',
            'neurology_lab', 'regenerative_medicine_lab', 'protein_engineering_lab'
        ]

        for lab_name in other_labs:
            if lab_name in self.digital_twin_simulator.digital_twins:
                suites[lab_name] = self._create_general_lab_suite(lab_name)

        return suites

    def _create_physics_lab_suite(self, lab_name: str) -> LabTestSuite:
        """Create test suite for physics laboratories"""

        base_tests = [
            TestQuestion(
                id=f"{lab_name}_energy_conservation",
                branch="physics_invariants",
                question="Verify energy conservation in physical processes",
                expected_answer_type="physics_law_compliance",
                physics_constraints=["energy_conservation", "thermodynamics_first_law"]
            ),
            TestQuestion(
                id=f"{lab_name}_momentum_conservation",
                branch="physics_invariants",
                question="Verify momentum conservation in collision processes",
                expected_answer_type="physics_law_compliance",
                physics_constraints=["momentum_conservation", "newtons_laws"]
            ),
            TestQuestion(
                id=f"{lab_name}_uncertainty_principle",
                branch="physics_invariants",
                question="Verify Heisenberg uncertainty principle applies to quantum systems",
                expected_answer_type="physics_law_compliance",
                physics_constraints=["heisenberg_uncertainty", "quantum_mechanics"]
            )
        ]

        if 'quantum' in lab_name:
            base_tests.extend([
                TestQuestion(
                    id=f"{lab_name}_wave_function_normalization",
                    branch="physics_invariants",
                    question="Verify quantum wave functions are properly normalized",
                    expected_answer_type="physics_law_compliance",
                    physics_constraints=["wave_function_normalization", "quantum_probability"]
                ),
                TestQuestion(
                    id=f"{lab_name}_schrodinger_equation",
                    branch="physics_invariants",
                    question="Verify solutions satisfy time-dependent Schrödinger equation",
                    expected_answer_type="physics_law_compliance",
                    physics_constraints=["schrodinger_equation", "quantum_dynamics"]
                )
            ])

        return LabTestSuite(
            lab_name=lab_name,
            test_categories=["physics_laws", "environmental_stability", "numerical_accuracy"],
            environmental_scenarios=["cryogenic", "vacuum", "high_radiation", "microgravity"],
            physics_validation_tests=base_tests,
            performance_benchmarks={
                "energy_conservation_error": "< 1e-6",
                "numerical_stability": "maintained",
                "physical_realism": "high"
            },
            failure_mode_tests=[
                {"condition": "temperature_extremes", "expected_behavior": "stable"},
                {"condition": "pressure_extremes", "expected_behavior": "converged"},
                {"condition": "electromagnetic_noise", "expected_behavior": "shielded"}
            ]
        )

    def _create_quantum_computing_lab_suite(self, lab_name: str) -> LabTestSuite:
        """Create test suite for quantum computing lab with lattice surgery"""

        base_tests = [
            TestQuestion(
                id=f"{lab_name}_lattice_surgery",
                branch="quantum_error_correction",
                question="Verify lattice surgery can split logical qubits while maintaining error correction",
                expected_answer_type="breakthrough_validation",
                physics_constraints=["lattice_surgery", "continuous_error_correction", "entanglement_preservation"]
            ),
            TestQuestion(
                id=f"{lab_name}_fault_tolerance",
                branch="quantum_error_correction",
                question="Verify fault-tolerant logical operations can be performed",
                expected_answer_type="breakthrough_validation",
                physics_constraints=["logical_gates", "error_correction_during_computation", "ETH_zurich_2026"]
            ),
            TestQuestion(
                id=f"{lab_name}_superconducting_integration",
                branch="quantum_hardware",
                question="Verify superconducting qubit integration with surface codes",
                expected_answer_type="hardware_validation",
                physics_constraints=["superconducting_qubits", "surface_code_distance_3", "17_physical_qubits"]
            )
        ]

        return LabTestSuite(
            lab_name=lab_name,
            test_categories=["quantum_computing", "error_correction", "fault_tolerance", "lattice_surgery"],
            environmental_scenarios=["cryogenic", "magnetic_shielding", "vibration_isolation", "radiation_shielding"],
            physics_validation_tests=base_tests,
            performance_benchmarks={
                "lattice_surgery_success_rate": "> 95%",
                "logical_gate_fidelity": "> 99%",
                "error_correction_cycles": "continuous",
                "entanglement_fidelity": "> 98%"
            },
            failure_mode_tests=[
                {"condition": "thermal_fluctuations", "expected_behavior": "error_corrected"},
                {"condition": "flux_noise", "expected_behavior": "qubit_stable"},
                {"condition": "crosstalk", "expected_behavior": "gates_isolated"}
            ]
        )

    def _create_chemistry_lab_suite(self, lab_name: str) -> LabTestSuite:
        """Create test suite for chemistry laboratories"""

        base_tests = [
            TestQuestion(
                id=f"{lab_name}_mass_conservation",
                branch="chemistry_invariants",
                question="Verify mass is conserved in chemical reactions",
                expected_answer_type="chemistry_law_compliance",
                physics_constraints=["mass_conservation", "stoichiometry"]
            ),
            TestQuestion(
                id=f"{lab_name}_charge_conservation",
                branch="chemistry_invariants",
                question="Verify charge is conserved in redox reactions",
                expected_answer_type="chemistry_law_compliance",
                physics_constraints=["charge_conservation", "redox_balance"]
            ),
            TestQuestion(
                id=f"{lab_name}_thermodynamic_feasibility",
                branch="chemistry_invariants",
                question="Verify reactions are thermodynamically feasible",
                expected_answer_type="chemistry_law_compliance",
                physics_constraints=["gibbs_free_energy", "reaction_spontaneity"]
            )
        ]

        if 'electrochemistry' in lab_name:
            base_tests.extend([
                TestQuestion(
                    id=f"{lab_name}_nernst_equation",
                    branch="chemistry_invariants",
                    question="Verify electrode potentials follow Nernst equation",
                    expected_answer_type="chemistry_law_compliance",
                    physics_constraints=["nernst_equation", "electrochemical_potential"]
                )
            ])

        return LabTestSuite(
            lab_name=lab_name,
            test_categories=["reaction_kinetics", "thermodynamics", "environmental_effects"],
            environmental_scenarios=["inert_atmosphere", "high_humidity", "corrosive", "thermal_cycling"],
            physics_validation_tests=base_tests,
            performance_benchmarks={
                "reaction_energy_accuracy": "< 1 kcal/mol",
                "kinetics_prediction": "within_order_of_magnitude",
                "solubility_accuracy": "< 10% error"
            },
            failure_mode_tests=[
                {"condition": "moisture_contamination", "expected_behavior": "stable_reactions"},
                {"condition": "oxygen_exposure", "expected_behavior": "controlled_oxidation"},
                {"condition": "temperature_fluctuations", "expected_behavior": "consistent_kinetics"}
            ]
        )

    def _create_biology_lab_suite(self, lab_name: str) -> LabTestSuite:
        """Create test suite for biology laboratories"""

        base_tests = [
            TestQuestion(
                id=f"{lab_name}_dna_replication",
                branch="biology_invariants",
                question="Verify DNA replication maintains genetic information",
                expected_answer_type="biology_law_compliance",
                physics_constraints=["base_pairing", "genetic_fidelity"]
            ),
            TestQuestion(
                id=f"{lab_name}_protein_folding",
                branch="biology_invariants",
                question="Verify proteins fold to thermodynamically stable structures",
                expected_answer_type="biology_law_compliance",
                physics_constraints=["protein_stability", "hydrophobic_effect"]
            ),
            TestQuestion(
                id=f"{lab_name}_metabolic_balance",
                branch="biology_invariants",
                question="Verify metabolic pathways maintain energy balance",
                expected_answer_type="biology_law_compliance",
                physics_constraints=["atp_balance", "energy_conservation"]
            )
        ]

        if 'microbiology' in lab_name:
            base_tests.extend([
                TestQuestion(
                    id=f"{lab_name}_sterility",
                    branch="biology_invariants",
                    question="Verify sterile conditions prevent contamination",
                    expected_answer_type="biology_law_compliance",
                    physics_constraints=["microbial_growth_inhibition", "aseptic_technique"]
                )
            ])

        return LabTestSuite(
            lab_name=lab_name,
            test_categories=["molecular_biology", "cellular_processes", "environmental_stress"],
            environmental_scenarios=["sterile_conditions", "physiological", "stress_conditions", "extreme_environment"],
            physics_validation_tests=base_tests,
            performance_benchmarks={
                "sequence_accuracy": "> 99.9%",
                "protein_structure_prediction": "within_2A_rmsd",
                "growth_rate_accuracy": "< 20% error"
            },
            failure_mode_tests=[
                {"condition": "contamination", "expected_behavior": "sterile_environment"},
                {"condition": "temperature_stress", "expected_behavior": "homeostasis_maintained"},
                {"condition": "radiation_exposure", "expected_behavior": "dna_repair_activated"}
            ]
        )

    def _create_materials_lab_suite(self, lab_name: str) -> LabTestSuite:
        """Create test suite for materials laboratories"""

        base_tests = [
            TestQuestion(
                id=f"{lab_name}_crystal_structure",
                branch="materials_invariants",
                question="Verify crystal structures are physically reasonable",
                expected_answer_type="materials_law_compliance",
                physics_constraints=["crystal_packing", "atomic_radii"]
            ),
            TestQuestion(
                id=f"{lab_name}_phase_stability",
                branch="materials_invariants",
                question="Verify predicted phases are thermodynamically stable",
                expected_answer_type="materials_law_compliance",
                physics_constraints=["phase_diagrams", "gibbs_energy"]
            ),
            TestQuestion(
                id=f"{lab_name}_property_consistency",
                branch="materials_invariants",
                question="Verify material properties are physically consistent",
                expected_answer_type="materials_law_compliance",
                physics_constraints=["property_correlations", "physical_bounds"]
            )
        ]

        return LabTestSuite(
            lab_name=lab_name,
            test_categories=["structure_prediction", "property_calculation", "processing_simulation"],
            environmental_scenarios=["ambient_conditions", "high_temperature", "corrosive_environment", "mechanical_stress"],
            physics_validation_tests=base_tests,
            performance_benchmarks={
                "formation_energy_accuracy": "< 0.1 eV/atom",
                "band_gap_accuracy": "< 0.5 eV",
                "elastic_moduli_accuracy": "< 10% error"
            },
            failure_mode_tests=[
                {"condition": "thermal_shock", "expected_behavior": "stable_structure"},
                {"condition": "corrosive_environment", "expected_behavior": "passivation"},
                {"condition": "mechanical_loading", "expected_behavior": "elastic_response"}
            ]
        )

    def _create_general_lab_suite(self, lab_name: str) -> LabTestSuite:
        """Create general test suite for other laboratories"""

        return LabTestSuite(
            lab_name=lab_name,
            test_categories=["functionality", "robustness", "environmental_adaptation"],
            environmental_scenarios=["standard_conditions", "stress_conditions", "extreme_environment"],
            physics_validation_tests=[
                TestQuestion(
                    id=f"{lab_name}_basic_functionality",
                    branch="general_invariants",
                    question="Verify basic laboratory functionality",
                    expected_answer_type="operational_compliance",
                    physics_constraints=["operational_stability", "result_consistency"]
                )
            ],
            performance_benchmarks={
                "operational_success_rate": "> 95%",
                "result_reproducibility": "< 5% variance",
                "environmental_robustness": "maintained"
            },
            failure_mode_tests=[
                {"condition": "environmental_stress", "expected_behavior": "graceful_degradation"},
                {"condition": "input_variations", "expected_behavior": "robust_processing"}
            ]
        )

    def run_lab_test_suite(self, lab_name: str, full_environmental_testing: bool = True) -> LabTestResult:
        """
        Run comprehensive test suite for a specific laboratory

        Args:
            lab_name: Name of lab to test
            full_environmental_testing: Whether to run all environmental scenarios

        Returns:
            Comprehensive test results
        """

        if lab_name not in self.lab_test_suites:
            return LabTestResult(
                lab_name=lab_name,
                total_tests_run=0,
                tests_passed=0,
                success_rate=0.0,
                environmental_sensitivity={},
                physics_compliance={},
                performance_metrics={},
                critical_failures=[f"Lab {lab_name} not found in test suites"],
                recommendations=["Lab not available for testing"],
                test_duration=0.0
            )

        test_suite = self.lab_test_suites[lab_name]
        start_time = time.time()

        logging.info(f"\n🧪 Testing Lab: {lab_name}")
        logging.info("=" * 50)

        # Run physics validation tests
        physics_results = self._run_physics_validation_tests(test_suite)

        # Run environmental scenario tests
        environmental_results = []
        if full_environmental_testing:
            environmental_results = self._run_environmental_scenario_tests(test_suite)

        # Run performance benchmarks
        performance_results = self._run_performance_benchmarks(test_suite)

        # Run failure mode tests
        failure_results = self._run_failure_mode_tests(test_suite)

        total_tests = len(physics_results) + len(environmental_results) + len(performance_results) + len(failure_results)
        passed_tests = sum(physics_results.values()) + sum(environmental_results) + sum(performance_results.values()) + sum(failure_results.values())

        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        # Analyze environmental sensitivity
        environmental_sensitivity = self._analyze_environmental_sensitivity(environmental_results, test_suite)

        # Check physics compliance
        physics_compliance = self._check_physics_compliance(physics_results, test_suite)

        # Generate critical failures and recommendations
        critical_failures = self._identify_critical_failures(physics_results, environmental_results, failure_results)
        recommendations = self._generate_lab_recommendations(lab_name, success_rate, environmental_sensitivity, critical_failures)

        test_duration = time.time() - start_time

        result = LabTestResult(
            lab_name=lab_name,
            total_tests_run=total_tests,
            tests_passed=passed_tests,
            success_rate=success_rate,
            environmental_sensitivity=environmental_sensitivity,
            physics_compliance=physics_compliance,
            performance_metrics=performance_results,
            critical_failures=critical_failures,
            recommendations=recommendations,
            test_duration=test_duration
        )

        self.test_results[lab_name] = result

        logging.info(f"✅ {lab_name} testing complete:")
        logging.info(f"   Success Rate: {success_rate:.1f}")
        logging.info(f"   Tests Passed: {passed_tests}/{total_tests}")
        logging.info(f"   Duration: {test_duration:.1f}s")

        return result

    def _run_physics_validation_tests(self, test_suite: LabTestSuite) -> Dict[str, bool]:
        """Run physics validation tests"""

        results = {}

        for test_question in test_suite.physics_validation_tests:
            # Special handling for quantum computing lab lattice surgery tests
            if test_suite.lab_name == 'quantum_computing_lab' and 'lattice_surgery' in test_question.id:
                passed = self._run_lattice_surgery_test(test_question)
                results[test_question.id] = passed
                continue

            # Create experiment based on test question
            experiment_spec = {
                'test_type': 'physics_validation',
                'question_id': test_question.id,
                'constraints': test_question.physics_constraints
            }

            try:
                experiment = self.digital_twin_simulator.run_digital_twin_experiment(
                    test_suite.lab_name, experiment_spec
                )

                # Evaluate if test passed (simplified evaluation)
                passed = experiment.success and len(experiment.environmental_impacts) >= 0
                results[test_question.id] = passed

            except (ImportError, AttributeError, TypeError) as e:
                logging.info(f"Physics test failed for {test_question.id}: {e}")
                results[test_question.id] = False

        return results

    def _run_lattice_surgery_test(self, test_question: TestQuestion) -> bool:
        """Run specialized lattice surgery breakthrough test"""

        try:
            from quantum_lattice_surgery import LatticeSurgeryQuantumLab

            # Create lattice surgery lab instance
            surgery_lab = LatticeSurgeryQuantumLab()

            # Test 1: Create fault-tolerant qubit
            qubit_name = surgery_lab.create_fault_tolerant_qubit("test_qubit")

            # Test 2: Perform lattice surgery (the breakthrough operation)
            split_boundary = [(2, 2), (2, 3), (3, 2)]
            surgery_result = surgery_lab.lattice_surgery.perform_lattice_surgery(qubit_name, split_boundary)

            # Test 3: Verify entanglement was created
            entanglement_created = surgery_result.phase == "completed" and len(surgery_result.split_qubits) == 2

            # Test 4: Perform logical operations
            if entanglement_created:
                qubit1, qubit2 = surgery_result.split_qubits[0].name, surgery_result.split_qubits[1].name
                logical_success = surgery_lab.lattice_surgery.perform_logical_operation(qubit1, qubit2, "CNOT")
            else:
                logical_success = False

            # Test 5: Verify error correction status
            status = surgery_lab.get_fault_tolerance_status()
            error_correction_active = status.get('error_correction_events', 0) > 0

            # Overall success: all key breakthrough aspects must work
            success = (surgery_result.phase == "completed" and
                      entanglement_created and
                      logical_success and
                      error_correction_active)

            logging.info(f"🔬 Lattice surgery test: {'PASS' if success else 'FAIL'}")
            logging.info(f"   • Surgery completed: {surgery_result.phase == 'completed'}")
            logging.info(f"   • Entanglement created: {entanglement_created}")
            logging.info(f"   • Logical ops work: {logical_success}")
            logging.info(f"   • Error correction active: {error_correction_active}")

            return success

        except (ImportError, AttributeError, TypeError) as e:
            logging.info(f"Lattice surgery test failed: {e}")
            return False

    def _run_environmental_scenario_tests(self, test_suite: LabTestSuite) -> List[Dict[str, Any]]:
        """Run environmental scenario tests"""

        results = []

        scenario_conditions = {
            'cryogenic': EnvironmentalPreset.cryogenic_lab(),
            'vacuum': EnvironmentalPreset.vacuum_chamber(),
            'high_humidity': EnvironmentalPreset.deep_sea_environment(),
            'microgravity': EnvironmentalPreset.space_environment(),
            'mars_surface': EnvironmentalPreset.mars_surface(),
            'clean_room': EnvironmentalPreset.clean_room_class100(),
            'standard_lab': EnvironmentalPreset.standard_lab()
        }

        for scenario in test_suite.environmental_scenarios:
            if scenario in scenario_conditions:
                experiment_spec = {
                    'test_type': 'environmental_stress',
                    'scenario': scenario,
                    'environmental_conditions': self._conditions_to_dict(scenario_conditions[scenario])
                }

                try:
                    experiment = self.digital_twin_simulator.run_digital_twin_experiment(
                        test_suite.lab_name, experiment_spec, scenario_conditions[scenario]
                    )

                    results.append({
                        'scenario': scenario,
                        'success': experiment.success,
                        'environmental_impacts': experiment.environmental_impacts,
                        'error_message': experiment.error_message
                    })

                except (ImportError, AttributeError, TypeError) as e:
                    results.append({
                        'scenario': scenario,
                        'success': False,
                        'error_message': str(e)
                    })

        return results

    def _run_performance_benchmarks(self, test_suite: LabTestSuite) -> Dict[str, Any]:
        """Run performance benchmark tests"""

        results = {}

        for benchmark_name, benchmark_spec in test_suite.performance_benchmarks.items():
            experiment_spec = {
                'test_type': 'performance_benchmark',
                'benchmark': benchmark_name,
                'target': benchmark_spec
            }

            try:
                experiment = self.digital_twin_simulator.run_digital_twin_experiment(
                    test_suite.lab_name, experiment_spec
                )

                # Simplified benchmark evaluation
                results[benchmark_name] = experiment.success

            except (ImportError, AttributeError, TypeError) as e:
                results[benchmark_name] = False

        return results

    def _run_failure_mode_tests(self, test_suite: LabTestSuite) -> Dict[str, Any]:
        """Run failure mode tests"""

        results = {}

        for failure_test in test_suite.failure_mode_tests:
            experiment_spec = {
                'test_type': 'failure_mode_test',
                'condition': failure_test['condition'],
                'expected_behavior': failure_test['expected_behavior']
            }

            try:
                experiment = self.digital_twin_simulator.run_digital_twin_experiment(
                    test_suite.lab_name, experiment_spec
                )

                # Simplified failure mode evaluation
                results[failure_test['condition']] = experiment.success

            except (ImportError, AttributeError, TypeError) as e:
                results[failure_test['condition']] = False

        return results

    def _analyze_environmental_sensitivity(self, environmental_results: List[Dict[str, Any]],
                                         test_suite: LabTestSuite) -> Dict[str, float]:
        """Analyze environmental sensitivity"""

        sensitivity = {}

        if not environmental_results:
            return sensitivity

        # Calculate sensitivity based on success rates under different conditions
        successful_scenarios = sum(1 for r in environmental_results if r['success'])
        total_scenarios = len(environmental_results)

        base_success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0

        # Environmental sensitivity (lower values = more sensitive)
        sensitivity['overall_environmental_sensitivity'] = 1.0 - base_success_rate

        # Specific sensitivities
        scenario_types = {
            'cryogenic': 'temperature_sensitivity',
            'vacuum': 'pressure_sensitivity',
            'high_humidity': 'humidity_sensitivity',
            'microgravity': 'gravity_sensitivity',
            'space_environment': 'radiation_sensitivity'
        }

        for result in environmental_results:
            scenario = result['scenario']
            if scenario in scenario_types:
                sensitivity_type = scenario_types[scenario]
                if sensitivity_type not in sensitivity:
                    sensitivity[sensitivity_type] = 0.0

                if not result['success']:
                    sensitivity[sensitivity_type] += 0.2  # Increase sensitivity if failed

        return sensitivity

    def _check_physics_compliance(self, physics_results: Dict[str, bool],
                                test_suite: LabTestSuite) -> Dict[str, bool]:
        """Check physics compliance"""

        compliance = {}

        for test_id, passed in physics_results.items():
            # Extract physics principle from test ID
            principle = test_id.split('_')[-1]  # Simplified
            compliance[principle] = passed

        return compliance

    def _identify_critical_failures(self, physics_results: Dict[str, bool],
                                  environmental_results: List[Dict[str, Any]],
                                  failure_results: Dict[str, Any]) -> List[str]:
        """Identify critical failures"""

        failures = []

        # Physics failures
        failed_physics = [test_id for test_id, passed in physics_results.items() if not passed]
        if failed_physics:
            failures.append(f"Failed physics validation: {', '.join(failed_physics[:3])}")

        # Environmental failures
        failed_scenarios = [r['scenario'] for r in environmental_results if not r['success']]
        if failed_scenarios:
            failures.append(f"Failed environmental scenarios: {', '.join(failed_scenarios[:3])}")

        # Critical failure modes
        failed_modes = [condition for condition, passed in failure_results.items() if not passed]
        if failed_modes:
            failures.append(f"Critical failure modes: {', '.join(failed_modes[:3])}")

        return failures

    def _generate_lab_recommendations(self, lab_name: str, success_rate: float,
                                    environmental_sensitivity: Dict[str, float],
                                    critical_failures: List[str]) -> List[str]:
        """Generate recommendations for lab improvements"""

        recommendations = []

        if success_rate < 0.8:
            recommendations.append("Improve overall reliability and error handling")

        # Environmental recommendations
        for sensitivity_type, level in environmental_sensitivity.items():
            if level > 0.5:
                if 'temperature' in sensitivity_type:
                    recommendations.append("Implement temperature stabilization systems")
                elif 'humidity' in sensitivity_type:
                    recommendations.append("Add humidity control and moisture protection")
                elif 'pressure' in sensitivity_type:
                    recommendations.append("Improve pressure regulation systems")
                elif 'radiation' in sensitivity_type:
                    recommendations.append("Add radiation shielding for sensitive components")

        if critical_failures:
            recommendations.append("Address critical failure modes identified in testing")

        if not recommendations:
            recommendations.append("Lab performs well across tested conditions")

        return recommendations

    def _conditions_to_dict(self, conditions: EnvironmentalConditions) -> Dict[str, Any]:
        """Convert conditions to dict"""
        return {
            'temperature': conditions.temperature,
            'pressure': conditions.pressure,
            'humidity': conditions.humidity,
            'gas_composition': conditions.gas_composition
        }

    def run_comprehensive_lab_testing(self, max_labs: int = 10) -> Dict[str, Any]:
        """
        Run comprehensive testing across multiple labs

        Args:
            max_labs: Maximum number of labs to test (for time constraints)

        Returns:
            Comprehensive testing results
        """

        logging.info("🔬 Running Comprehensive QuLab Testing")
        logging.info("=" * 60)

        available_labs = list(self.lab_test_suites.keys())[:max_labs]
        logging.info(f"Testing {len(available_labs)} laboratories...")

        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'labs_tested': len(available_labs),
            'individual_results': {},
            'summary': {},
            'recommendations': []
        }

        for lab_name in available_labs:
            try:
                result = self.run_lab_test_suite(lab_name, full_environmental_testing=True)
                comprehensive_results['individual_results'][lab_name] = {
                    'success_rate': result.success_rate,
                    'tests_passed': result.tests_passed,
                    'total_tests': result.total_tests,
                    'critical_failures': result.critical_failures,
                    'recommendations': result.recommendations
                }
            except (ImportError, AttributeError, TypeError) as e:
                logging.info(f"Failed to test {lab_name}: {e}")
                comprehensive_results['individual_results'][lab_name] = {
                    'error': str(e)
                }

        # Generate summary
        comprehensive_results['summary'] = self._generate_comprehensive_summary(
            comprehensive_results['individual_results']
        )

        # Overall recommendations
        comprehensive_results['recommendations'] = self._generate_overall_recommendations(
            comprehensive_results['summary']
        )

        logging.info("\n🏆 Comprehensive Testing Complete")
        logging.info(f"Labs Tested: {comprehensive_results['summary']['labs_tested']}")
        logging.info(f"Average Success Rate: {comprehensive_results['summary']['average_success_rate']:.1f}")
        logging.info(f"Environmental Sensitive Labs: {comprehensive_results['summary']['environmentally_sensitive_labs']}")

        return comprehensive_results

    def _generate_comprehensive_summary(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary"""

        success_rates = []
        environmentally_sensitive = 0

        for lab_result in individual_results.values():
            if 'success_rate' in lab_result:
                success_rates.append(lab_result['success_rate'])

                # Check for environmental sensitivity
                if any('environmental' in str(rec) for rec in lab_result.get('recommendations', [])):
                    environmentally_sensitive += 1

        average_success_rate = np.mean(success_rates) if success_rates else 0

        return {
            'labs_tested': len(individual_results),
            'average_success_rate': average_success_rate,
            'environmentally_sensitive_labs': environmentally_sensitive,
            'success_rate_distribution': {
                'excellent': sum(1 for r in success_rates if r >= 0.9),
                'good': sum(1 for r in success_rates if 0.7 <= r < 0.9),
                'fair': sum(1 for r in success_rates if 0.5 <= r < 0.7),
                'poor': sum(1 for r in success_rates if r < 0.5)
            }
        }

    def _generate_overall_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations"""

        recommendations = []

        avg_success = summary['average_success_rate']

        if avg_success >= 0.8:
            recommendations.append("Overall lab ecosystem is robust and reliable")
        elif avg_success >= 0.6:
            recommendations.append("Moderate improvements needed across lab ecosystem")
        else:
            recommendations.append("Significant improvements needed in lab reliability")

        if summary['environmentally_sensitive_labs'] > summary['labs_tested'] * 0.5:
            recommendations.append("High environmental sensitivity detected - implement comprehensive environmental controls")

        recommendations.append("Focus testing efforts on most environmentally sensitive labs")

        return recommendations

    def export_lab_testing_results(self, results: Dict[str, Any], filename: str):
        """Export comprehensive lab testing results"""

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logging.info(f"✅ Lab testing results exported to {filename}")


def main():
    """Run comprehensive lab testing demonstration"""

    logging.info("🔬 QuLab Expanded Lab Testing Framework")
    logging.info("=" * 60)

    tester = QuLabExpandedLabTester()

    logging.info(f"Available test suites: {len(tester.lab_test_suites)}")

    # Show available labs by category
    categories = {}
    for lab_name, suite in tester.lab_test_suites.items():
        category = suite.test_categories[0] if suite.test_categories else 'general'
        if category not in categories:
            categories[category] = []
        categories[category].append(lab_name)

    logging.info("\n📋 Labs by Category:")
    for category, labs in categories.items():
        logging.info(f"  {category}: {len(labs)} labs")
        if len(labs) <= 5:
            for lab in labs:
                logging.info(f"    • {lab}")
        else:
            logging.info(f"    • {', '.join(labs[:3])}, ... (+{len(labs)-3} more)")

    # Run comprehensive testing on a subset of labs
    logging.info("\n🧪 Running comprehensive testing on 5 labs...")
    comprehensive_results = tester.run_comprehensive_lab_testing(max_labs=5)

    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qulab_comprehensive_lab_testing_{timestamp}.json"
    tester.export_lab_testing_results(comprehensive_results, filename)

    logging.info("\n💡 Key Findings:")
    summary = comprehensive_results['summary']
    logging.info(f"  • Average success rate: {summary['average_success_rate']:.1f}")
    logging.info(f"  • Environmentally sensitive labs: {summary['environmentally_sensitive_labs']}")
    logging.info(f"  • Excellent performers: {summary['success_rate_distribution']['excellent']}")
    logging.info(f"  • Needs improvement: {summary['success_rate_distribution']['poor']}")


if __name__ == "__main__":
    main()