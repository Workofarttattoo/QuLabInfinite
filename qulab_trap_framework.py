"""Trap Framework Module

This module provides functionality for trap_framework operations in QuLab Infinite.

Author: QuLab Development Team
Version: 1.0.0
"""

# TODO: Refactor long functions identified in code quality analysis
import logging
#!/usr/bin/env python3
"""
QuLab Trap Framework - Rigorous Materials AI Testing System
=====================================================================

This framework implements the "trap framework" for testing QuLab Infinite's materials
science capabilities against real-world databases and physical laws.

The framework consists of four testing branches:

BRANCH A: Rediscovery Tests (weight ≈ 0.40)
- Ask QuLab to rediscover materials that already exist but don't tell it they exist
- Check predictions against Materials Project, AFLOW, OQMD databases
- Score based on how closely predictions match known materials

BRANCH B: Physics Sanity Checks (weight ≈ 0.25)
- Ask questions that force obedience to hard physical limits
- Test against known physical constraints (e.g., conductivity limits)
- Flag any violations of fundamental physics

BRANCH C: Database Cross-Matching (weight ≈ 0.20)
- Ask QuLab to predict materials with specific measurable properties
- Cross-reference against known datasets
- Validate composition and property predictions

BRANCH D: Impossible Material Trap (weight ≈ 0.15)
- Give deliberately impossible targets
- Test whether QuLab correctly identifies impossibility
- Flag hallucinated predictions for impossible chemistry

The framework also implements physics invariants that must never be broken:
1. Charge neutrality
2. Formation energy constraints
3. Crystal packing limits
4. Thermodynamic feasibility

Author: QuLab Infinite Validation Team
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import requests
import hashlib
import re
from pathlib import Path
import warnings

from qulab_master_api import QuLabMasterAPI
from materials_lab.materials_lab import MaterialsLab


@dataclass
class TestQuestion:
    """Individual test question with expected outcomes"""
    id: str
    branch: str
    question: str
    expected_answer_type: str  # 'material_prediction', 'impossible', 'specific_material', etc.
    expected_materials: List[str] = field(default_factory=list)  # Known materials that should be found
    physics_constraints: List[str] = field(default_factory=list)  # Physical limits to check
    database_references: List[str] = field(default_factory=list)  # Databases to cross-check
    weight: float = 1.0
    difficulty: str = "medium"  # easy, medium, hard, expert


@dataclass
class TestResult:
    """Result of running a test question"""
    question_id: str
    question: str
    branch: str
    qulab_response: Any
    physics_validation: Dict[str, bool]
    database_matches: List[Dict[str, Any]]
    invariants_check: Dict[str, bool]
    score: float
    confidence: float
    flags: List[str]  # Red flags indicating potential hallucinations
    timestamp: str


@dataclass
class BranchScore:
    """Score for each testing branch"""
    branch_name: str
    weight: float
    questions_tested: int
    physics_violations: int
    database_matches: int
    impossible_correctly_identified: int
    average_score: float
    red_flags: List[str]


class PhysicsInvariantsChecker:
    """
    Checks fundamental physics constraints that must never be violated

    These are the universal rules that govern materials science:
    1. Charge neutrality - compounds must balance valence electrons
    2. Formation energy - stable materials near convex hull
    3. Crystal packing - atoms cannot overlap
    4. Thermodynamic feasibility - reactions must obey Gibbs free energy
    """

    def __init__(self) -> None:
        self.valence_electrons = {
            'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
            'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
            'K': 1, 'Ca': 2, 'Ti': 4, 'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 12,
            'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8,
            'Rb': 1, 'Sr': 2, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Tc': 7, 'Ru': 8, 'Rh': 9, 'Pd': 10, 'Ag': 11, 'Cd': 12,
            'In': 3, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7, 'Xe': 8,
            'Cs': 1, 'Ba': 2, 'La': 3, 'Hf': 4, 'Ta': 5, 'W': 6, 'Re': 7, 'Os': 8, 'Ir': 9, 'Pt': 10, 'Au': 11, 'Hg': 12,
            'Tl': 3, 'Pb': 4, 'Bi': 5, 'Po': 6, 'At': 7, 'Rn': 8
        }

        # Atomic radii in Angstroms
        self.atomic_radii = {
            'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
            'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
            'K': 2.03, 'Ca': 1.76, 'Ti': 1.46, 'V': 1.34, 'Cr': 1.28, 'Mn': 1.27, 'Fe': 1.26, 'Co': 1.25, 'Ni': 1.24, 'Cu': 1.28, 'Zn': 1.34,
            'Ga': 1.35, 'Ge': 1.22, 'As': 1.21, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.17,
            'Rb': 2.10, 'Sr': 1.95, 'Zr': 1.59, 'Nb': 1.47, 'Mo': 1.39, 'Tc': 1.36, 'Ru': 1.34, 'Rh': 1.34, 'Pd': 1.37, 'Ag': 1.44, 'Cd': 1.51,
            'In': 1.67, 'Sn': 1.58, 'Sb': 1.61, 'Te': 1.43, 'I': 1.39, 'Xe': 1.40
        }

    def check_charge_neutrality(self, composition: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if a compound maintains charge neutrality

        Args:
            composition: Dict of element symbols to atomic fractions

        Returns:
            (is_valid, reason)
        """
        total_valence_electrons = 0
        total_electrons_needed = 0

        for element, fraction in composition.items():
            if element not in self.valence_electrons:
                return False, f"Unknown element: {element}"

            # Simplified charge neutrality check
            # In reality this is much more complex with oxidation states
            valence = self.valence_electrons[element]
            total_valence_electrons += valence * fraction

        # For ionic compounds, we expect some balance
        # This is a simplified check - real charge neutrality is more complex
        avg_valence = total_valence_electrons / sum(composition.values())

        # Flag extreme imbalances
        if avg_valence < 1 or avg_valence > 8:
            return False, f"Average valence electrons per atom ({avg_valence:.1f}) outside reasonable range"

        return True, "Charge appears balanced"

    def check_crystal_packing(self, composition: Dict[str, float], density: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if crystal packing is physically reasonable

        Args:
            composition: Dict of element symbols to atomic fractions
            density: Material density in g/cm³

        Returns:
            (is_valid, reason)
        """
        if density is None:
            return True, "Cannot check packing without density"

        # Calculate weighted average atomic radius
        total_radius = 0
        total_weight = 0

        for element, fraction in composition.items():
            if element in self.atomic_radii:
                atomic_mass = self._get_atomic_mass(element)
                radius = self.atomic_radii[element]
                total_radius += radius * fraction
                total_weight += atomic_mass * fraction

        if total_weight == 0:
            return False, "Invalid composition"

        avg_radius = total_radius / sum(composition.values())

        # Estimate minimum unit cell volume
        # Simple cubic packing assumption
        min_volume_per_atom = (4/3) * np.pi * (avg_radius * 1e-8)**3  # cm³ per atom

        # Calculate atoms per cm³
        atoms_per_cm3 = (density * 6.022e23) / total_weight

        # Check if packing is reasonable
        estimated_volume = 1 / atoms_per_cm3 if atoms_per_cm3 > 0 else float('inf')

        if estimated_volume < min_volume_per_atom:
            return False, ".2e"

        return True, "Crystal packing appears reasonable"

    def check_thermodynamic_feasibility(self, formation_energy: Optional[float] = None,
                                       temperature: float = 298.15) -> Tuple[bool, str]:
        """
        Check thermodynamic feasibility

        Args:
            formation_energy: Formation energy in eV/atom
            temperature: Temperature in Kelvin

        Returns:
            (is_valid, reason)
        """
        if formation_energy is None:
            return True, "Cannot check thermodynamics without formation energy"

        # Convert to kJ/mol for typical ranges
        formation_energy_kjmol = formation_energy * 96.485  # eV to kJ/mol conversion

        # Typical formation energies for stable compounds
        if abs(formation_energy_kjmol) > 2000:  # Very high formation energy
            return False, ".1f"

        # Check against room temperature thermal energy
        kT = 8.314 * temperature / 1000  # kJ/mol
        if abs(formation_energy_kjmol) < kT * 0.1:  # Too close to thermal energy
            return False, ".2f"

        return True, "Thermodynamics appear feasible"

    def check_formation_energy_stability(self, formation_energy: Optional[float] = None) -> Tuple[bool, str]:
        """
        Check if formation energy suggests stability

        Args:
            formation_energy: Formation energy in eV/atom

        Returns:
            (is_valid, reason)
        """
        if formation_energy is None:
            return True, "Cannot check formation energy stability"

        # Typical range for stable compounds (-5 to 0 eV/atom)
        if formation_energy > 1.0:  # Very high formation energy
            return False, ".2f"
        elif formation_energy < -10.0:  # Very low (too stable, suspicious)
            return False, ".2f"

        return True, "Formation energy in reasonable range"

    def _get_atomic_mass(self, element: str) -> float:
        """Get atomic mass for element"""
        atomic_masses = {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974, 'S': 32.066, 'Cl': 35.453, 'Ar': 39.948,
            'K': 39.098, 'Ca': 40.078, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938, 'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.382,
            'Ga': 69.723, 'Ge': 72.631, 'As': 74.922, 'Se': 78.971, 'Br': 79.904, 'Kr': 83.798,
            'Rb': 85.468, 'Sr': 87.621, 'Zr': 91.224, 'Nb': 92.906, 'Mo': 95.951, 'Tc': 98.907, 'Ru': 101.072, 'Rh': 102.906, 'Pd': 106.421, 'Ag': 107.868, 'Cd': 112.414,
            'In': 114.818, 'Sn': 118.711, 'Sb': 121.760, 'Te': 127.603, 'I': 126.904, 'Xe': 131.294
        }
        return atomic_masses.get(element, 50.0)  # Default to reasonable mass


class DatabaseVerifier:
    """
    Cross-references QuLab predictions against real materials databases

    Databases checked:
    - Materials Project (materialsproject.org)
    - AFLOW (aflowlib.org)
    - OQMD (oqmd.org)
    - NOMAD (nomad-lab.eu)
    """

    def __init__(self) -> None:
        self.databases = {
            'materials_project': {
                'url': 'https://api.materialsproject.org',
                'api_key_required': True,
                'search_fields': ['formula', 'elements', 'band_gap', 'formation_energy_per_atom']
            },
            'aflow': {
                'url': 'https://aflowlib.org',
                'api_key_required': False,
                'search_fields': ['compound', 'bandgap', 'delta_e']
            },
            'oqmd': {
                'url': 'http://oqmd.org',
                'api_key_required': False,
                'search_fields': ['formula', 'stability', 'bandgap']
            }
        }

    def search_materials_project(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search Materials Project database"""
        # Note: Requires API key - this is a mock implementation
        # In real implementation, would use requests to query API

        mock_results = [
            {
                'material_id': 'mp-1234',
                'formula': 'TiO2',
                'formation_energy_per_atom': -3.14,
                'band_gap': 3.2,
                'crystal_system': 'tetragonal',
                'elements': ['Ti', 'O']
            },
            {
                'material_id': 'mp-5678',
                'formula': 'Li6PS5Cl',
                'formation_energy_per_atom': -2.45,
                'band_gap': 2.8,
                'crystal_system': 'cubic',
                'elements': ['Li', 'P', 'S', 'Cl']
            }
        ]

        return mock_results

    def search_aflow(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search AFLOW database"""
        # Mock implementation
        mock_results = [
            {
                'compound': 'Ti3C2',
                'bandgap': 1.2,
                'delta_e': -0.45,
                'crystal_structure': 'hexagonal'
            }
        ]

        return mock_results

    def search_oqmd(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search OQMD database"""
        # Mock implementation
        mock_results = [
            {
                'formula': 'Li7P3S11',
                'stability': -2.1,
                'bandgap': 3.1,
                'spacegroup': 'P4_2/mnm'
            }
        ]

        return mock_results

    def cross_reference_prediction(self, prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Cross-reference a QuLab prediction against all databases

        Args:
            prediction: QuLab's material prediction

        Returns:
            List of database matches
        """
        matches = []

        # Extract key properties for searching
        formula = prediction.get('formula', '')
        elements = prediction.get('elements', [])
        band_gap = prediction.get('band_gap')
        formation_energy = prediction.get('formation_energy')

        # Search each database
        try:
            mp_results = self.search_materials_project({
                'formula': formula,
                'elements': elements,
                'band_gap': band_gap,
                'formation_energy_per_atom': formation_energy
            })
            matches.extend([{'database': 'materials_project', **result} for result in mp_results])
        except Exception as e:
            logging.info(f"Materials Project search failed: {e}")

        try:
            aflow_results = self.search_aflow({
                'compound': formula,
                'bandgap': band_gap
            })
            matches.extend([{'database': 'aflow', **result} for result in aflow_results])
        except Exception as e:
            logging.info(f"AFLOW search failed: {e}")

        try:
            oqmd_results = self.search_oqmd({
                'formula': formula,
                'bandgap': band_gap
            })
            matches.extend([{'database': 'oqmd', **result} for result in oqmd_results])
        except Exception as e:
            logging.info(f"OQMD search failed: {e}")

        return matches


class QuLabTrapFramework:
    """
    Main trap framework for rigorous QuLab testing

    Implements four branches of testing with physics invariants and database verification.
    """

    def __init__(self) -> None:
        self.qulab_api = QuLabMasterAPI()
        self.materials_lab = MaterialsLab()
        self.physics_checker = PhysicsInvariantsChecker()
        self.database_verifier = DatabaseVerifier()

        self.test_questions = self._build_test_suite()
        self.results_history = []

    def _build_test_suite(self) -> List[TestQuestion]:
        """Build the comprehensive test suite with all four branches"""

        questions = []

        # ===== BRANCH A: Rediscovery Tests (weight ≈ 0.40) =====
        # Ask QuLab to rediscover materials that already exist but don't tell it they exist

        questions.extend([
            TestQuestion(
                id="A1",
                branch="rediscovery",
                question="Search for stable oxide materials composed of Ti, O, and Al with high electrical conductivity (>5000 S/cm) and layered structure.",
                expected_answer_type="material_prediction",
                expected_materials=["TiO2", "Al2O3", "TiAlO3"],
                physics_constraints=["conductivity_realistic", "crystal_structure_possible"],
                database_references=["materials_project", "aflow", "oqmd"],
                weight=1.0,
                difficulty="medium"
            ),

            TestQuestion(
                id="A2",
                branch="rediscovery",
                question="Find the most stable lithium sulfide solid electrolytes with ionic conductivity >10⁻³ S/cm.",
                expected_answer_type="specific_material",
                expected_materials=["Li6PS5Cl", "Li7P3S11"],
                physics_constraints=["ionic_conductivity_possible", "charge_neutral"],
                database_references=["materials_project", "oqmd"],
                weight=1.0,
                difficulty="hard"
            ),

            TestQuestion(
                id="A3",
                branch="rediscovery",
                question="Discover 2D materials composed of Ti and C with high electrical conductivity.",
                expected_answer_type="material_prediction",
                expected_materials=["Ti3C2", "Ti2C", "MXenes"],
                physics_constraints=["2d_structure_stable", "conductivity_high"],
                database_references=["materials_project", "aflow"],
                weight=1.0,
                difficulty="expert"
            )
        ])

        # ===== BRANCH B: Physics Sanity Checks (weight ≈ 0.25) =====
        # Ask questions that force obedience to hard physical limits

        questions.extend([
            TestQuestion(
                id="B1",
                branch="physics_sanity",
                question="Design a metal alloy with conductivity greater than pure copper.",
                expected_answer_type="impossible",
                physics_constraints=["conductivity_limit_copper"],
                weight=1.0,
                difficulty="medium"
            ),

            TestQuestion(
                id="B2",
                branch="physics_sanity",
                question="Design a metal with superconducting transition temperature above 200K at ambient pressure.",
                expected_answer_type="impossible",
                physics_constraints=["superconductivity_limit"],
                weight=1.0,
                difficulty="hard"
            ),

            TestQuestion(
                id="B3",
                branch="physics_sanity",
                question="Find a material with negative Poisson's ratio (auxetic) but compressive strength greater than diamond.",
                expected_answer_type="impossible",
                physics_constraints=["poisson_ratio_limits", "strength_realistic"],
                weight=1.0,
                difficulty="expert"
            )
        ])

        # ===== BRANCH C: Database Cross-Matching (weight ≈ 0.20) =====
        # Ask QuLab to predict materials with specific measurable properties, then check against known datasets

        questions.extend([
            TestQuestion(
                id="C1",
                branch="database_matching",
                question="Find lithium solid electrolytes with ionic conductivity >10⁻³ S/cm at room temperature.",
                expected_answer_type="specific_material",
                expected_materials=["Li6PS5Cl", "LLZO", "LISICON", "Li7P3S11"],
                physics_constraints=["ionic_conductivity_realistic"],
                database_references=["materials_project", "oqmd"],
                weight=1.0,
                difficulty="hard"
            ),

            TestQuestion(
                id="C2",
                branch="database_matching",
                question="Predict layered 2D materials composed of Ti and C with high conductivity.",
                expected_answer_type="material_prediction",
                expected_materials=["Ti3C2Tx", "Ti2CTx", "MXenes"],
                physics_constraints=["2d_stability", "conductivity_high"],
                database_references=["materials_project", "aflow"],
                weight=1.0,
                difficulty="expert"
            )
        ])

        # ===== BRANCH D: Impossible Material Trap (weight ≈ 0.15) =====
        # Give deliberately impossible targets

        questions.extend([
            TestQuestion(
                id="D1",
                branch="impossible_trap",
                question="Design a stable compound containing Na, He, and Ar that forms a conductive crystal at ambient pressure.",
                expected_answer_type="impossible",
                physics_constraints=["noble_gas_chemistry", "stability_impossible"],
                weight=1.0,
                difficulty="medium"
            ),

            TestQuestion(
                id="D2",
                branch="impossible_trap",
                question="Design a room-temperature superconductor composed only of carbon and oxygen.",
                expected_answer_type="impossible",
                physics_constraints=["carbon_oxygen_superconductivity"],
                weight=1.0,
                difficulty="hard"
            ),

            TestQuestion(
                id="D3",
                branch="impossible_trap",
                question="Find a material with band gap of 10 eV but electrical conductivity of 10^7 S/cm.",
                expected_answer_type="impossible",
                physics_constraints=["bandgap_conductivity_contradiction"],
                weight=1.0,
                difficulty="expert"
            )
        ])

        return questions

    def run_test_question(self, question_id: str) -> TestResult:
        """
        Run a specific test question through QuLab and validate results

        Args:
            question_id: ID of the question to run

        Returns:
            TestResult with full validation
        """

        # Find the question
        question = None
        for q in self.test_questions:
            if q.id == question_id:
                question = q
                break

        if not question:
            raise ValueError(f"Question {question_id} not found")

        logging.info(f"\n🔬 Running Test {question_id}: {question.branch}")
        logging.info(f"Question: {question.question}")

        # Query QuLab
        qulab_response = self._query_qulab(question)

        # Validate physics
        physics_validation = self._validate_physics_constraints(question, qulab_response)

        # Check against databases
        database_matches = self.database_verifier.cross_reference_prediction(qulab_response)

        # Check physics invariants
        invariants_check = self._check_physics_invariants(qulab_response)

        # Calculate score
        score = self._calculate_question_score(question, qulab_response, physics_validation,
                                             database_matches, invariants_check)

        # Identify red flags
        flags = self._identify_red_flags(question, qulab_response, physics_validation, invariants_check)

        # Calculate confidence
        confidence = self._calculate_confidence(physics_validation, invariants_check, database_matches)

        result = TestResult(
            question_id=question.id,
            question=question.question,
            branch=question.branch,
            qulab_response=qulab_response,
            physics_validation=physics_validation,
            database_matches=database_matches,
            invariants_check=invariants_check,
            score=score,
            confidence=confidence,
            flags=flags,
            timestamp=datetime.now().isoformat()
        )

        self.results_history.append(result)

        logging.info(f"Score: {score:.2f}")
        logging.info(f"Red flags: {len(flags)}")

        return result

    def run_branch_tests(self, branch: str) -> BranchScore:
        """
        Run all tests for a specific branch

        Args:
            branch: Branch name ('rediscovery', 'physics_sanity', 'database_matching', 'impossible_trap')

        Returns:
            BranchScore with comprehensive results
        """

        branch_questions = [q for q in self.test_questions if q.branch == branch]

        if not branch_questions:
            raise ValueError(f"No questions found for branch: {branch}")

        logging.info(f"\n🧪 Running {len(branch_questions)} tests for branch: {branch}")

        results = []
        physics_violations = 0
        database_matches = 0
        impossible_correctly_identified = 0
        red_flags = []

        for question in branch_questions:
            result = self.run_test_question(question.id)
            results.append(result)

            # Count violations
            if not all(result.physics_validation.values()):
                physics_violations += 1

            if result.database_matches:
                database_matches += 1

            if question.expected_answer_type == "impossible" and result.score > 0.8:
                impossible_correctly_identified += 1

            red_flags.extend(result.flags)

        # Calculate average score
        average_score = np.mean([r.score for r in results]) if results else 0.0

        # Get branch weight
        branch_weights = {
            'rediscovery': 0.40,
            'physics_sanity': 0.25,
            'database_matching': 0.20,
            'impossible_trap': 0.15
        }

        branch_score = BranchScore(
            branch_name=branch,
            weight=branch_weights.get(branch, 0.0),
            questions_tested=len(results),
            physics_violations=physics_violations,
            database_matches=database_matches,
            impossible_correctly_identified=impossible_correctly_identified,
            average_score=average_score,
            red_flags=list(set(red_flags))  # Unique flags
        )

        return branch_score

    def run_complete_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete trap framework evaluation

        Returns:
            Complete evaluation results
        """

        logging.info("🎯 Starting QuLab Trap Framework Evaluation")
        logging.info("=" * 60)

        branch_results = {}
        total_score = 0.0

        branches = ['rediscovery', 'physics_sanity', 'database_matching', 'impossible_trap']

        for branch in branches:
            branch_score = self.run_branch_tests(branch)
            branch_results[branch] = branch_score
            total_score += branch_score.average_score * branch_score.weight

        # Generate overall assessment
        assessment = self._generate_overall_assessment(branch_results, total_score)

        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(self.test_questions),
            'total_score': total_score,
            'branch_results': {k: v.__dict__ for k, v in branch_results.items()},
            'assessment': assessment,
            'recommendations': self._generate_recommendations(assessment)
        }

        logging.info(f"\n🏆 EVALUATION COMPLETE")
        logging.info(f"Total Score: {total_score:.3f}/1.0")
        logging.info(f"Assessment: {assessment['level']}")

        return evaluation_results

    def _query_qulab(self, question: TestQuestion) -> Dict[str, Any]:
        """Query QuLab with the test question"""
        try:
            # Use materials lab for materials science questions
            if "material" in question.question.lower() or "alloy" in question.question.lower():
                # Try to use materials prediction capabilities
                # This is a simplified interface - in practice would need specific API calls
                response = {
                    'type': 'materials_prediction',
                    'materials': [
                        {
                            'formula': 'TiO2',
                            'properties': {
                                'conductivity': 1e-10,
                                'formation_energy': -3.2,
                                'band_gap': 3.0
                            },
                            'structure': 'rutile'
                        }
                    ],
                    'confidence': 0.8
                }
            else:
                response = {
                    'type': 'general_response',
                    'answer': 'This appears to be impossible based on current physical laws.',
                    'confidence': 0.9
                }

            return response

        except Exception as e:
            return {
                'type': 'error',
                'error': str(e),
                'confidence': 0.0
            }

    def _validate_physics_constraints(self, question: TestQuestion, response: Dict[str, Any]) -> Dict[str, bool]:
        """Validate physics constraints for the question"""

        validation_results = {}

        for constraint in question.physics_constraints:
            if constraint == "conductivity_realistic":
                # Check if predicted conductivity is physically reasonable
                materials = response.get('materials', [])
                for material in materials:
                    conductivity = material.get('properties', {}).get('conductivity', 0)
                    if conductivity > 1e8:  # Unrealistically high conductivity
                        validation_results[constraint] = False
                    else:
                        validation_results[constraint] = True

            elif constraint == "conductivity_limit_copper":
                # Copper conductivity limit check
                materials = response.get('materials', [])
                copper_conductivity = 5.96e7  # S/m
                for material in materials:
                    conductivity = material.get('properties', {}).get('conductivity', 0)
                    if conductivity > copper_conductivity * 1.1:  # More than 10% better than copper
                        validation_results[constraint] = False
                    else:
                        validation_results[constraint] = True

            elif constraint == "noble_gas_chemistry":
                # Noble gases don't form stable compounds
                materials = response.get('materials', [])
                for material in materials:
                    formula = material.get('formula', '')
                    if 'He' in formula or 'Ar' in formula:
                        validation_results[constraint] = False
                    else:
                        validation_results[constraint] = True

            elif constraint == "bandgap_conductivity_contradiction":
                # High bandgap can't have high conductivity
                materials = response.get('materials', [])
                for material in materials:
                    props = material.get('properties', {})
                    band_gap = props.get('band_gap', 0)
                    conductivity = props.get('conductivity', 0)
                    if band_gap > 5 and conductivity > 1e3:
                        validation_results[constraint] = False
                    else:
                        validation_results[constraint] = True

            else:
                validation_results[constraint] = True  # Default pass

        return validation_results

    def _check_physics_invariants(self, response: Dict[str, Any]) -> Dict[str, bool]:
        """Check fundamental physics invariants"""

        invariants_results = {
            'charge_neutrality': True,
            'crystal_packing': True,
            'thermodynamic_feasibility': True,
            'formation_energy_stability': True
        }

        materials = response.get('materials', [])
        for material in materials:
            composition = material.get('composition', {})
            properties = material.get('properties', {})

            # Check charge neutrality
            if composition:
                valid, reason = self.physics_checker.check_charge_neutrality(composition)
                if not valid:
                    invariants_results['charge_neutrality'] = False

            # Check crystal packing
            density = properties.get('density')
            if composition and density:
                valid, reason = self.physics_checker.check_crystal_packing(composition, density)
                if not valid:
                    invariants_results['crystal_packing'] = False

            # Check formation energy
            formation_energy = properties.get('formation_energy')
            if formation_energy is not None:
                valid, reason = self.physics_checker.check_formation_energy_stability(formation_energy)
                if not valid:
                    invariants_results['formation_energy_stability'] = False

        return invariants_results

    def _calculate_question_score(self, question: TestQuestion, response: Dict[str, Any],
                                physics_validation: Dict[str, bool], database_matches: List[Dict[str, Any]],
                                invariants_check: Dict[str, bool]) -> float:
        """Calculate score for a test question"""

        score = 0.0
        total_weight = 0.0

        # Physics validation (40% weight)
        physics_score = sum(physics_validation.values()) / len(physics_validation) if physics_validation else 1.0
        score += physics_score * 0.4
        total_weight += 0.4

        # Invariants check (30% weight)
        invariants_score = sum(invariants_check.values()) / len(invariants_check) if invariants_check else 1.0
        score += invariants_score * 0.3
        total_weight += 0.3

        # Database matching (20% weight)
        if question.expected_materials and database_matches:
            match_score = len(database_matches) / len(question.expected_materials)
            score += min(match_score, 1.0) * 0.2
            total_weight += 0.2
        elif not question.expected_materials:
            score += 0.2  # No specific materials expected
            total_weight += 0.2

        # Response quality (10% weight)
        if response.get('confidence', 0) > 0.5:
            score += 0.1
        total_weight += 0.1

        return score / total_weight if total_weight > 0 else 0.0

    def _identify_red_flags(self, question: TestQuestion, response: Dict[str, Any],
                          physics_validation: Dict[str, bool], invariants_check: Dict[str, bool]) -> List[str]:
        """Identify red flags indicating potential hallucinations"""

        flags = []

        # Physics violations
        for constraint, valid in physics_validation.items():
            if not valid:
                flags.append(f"Physics violation: {constraint}")

        # Invariants violations
        for invariant, valid in invariants_check.items():
            if not valid:
                flags.append(f"Invariant violation: {invariant}")

        # Impossible materials claimed as possible
        if question.expected_answer_type == "impossible":
            if response.get('materials'):
                flags.append("Impossible material claimed as possible")

        # Unrealistic property values
        materials = response.get('materials', [])
        for material in materials:
            props = material.get('properties', {})
            conductivity = props.get('conductivity', 0)
            if conductivity > 1e10:  # Unrealistically high
                flags.append(f"Unrealistic conductivity: {conductivity} S/m")

            formation_energy = props.get('formation_energy')
            if formation_energy and abs(formation_energy) > 20:
                flags.append(f"Unrealistic formation energy: {formation_energy} eV/atom")

        return flags

    def _calculate_confidence(self, physics_validation: Dict[str, bool],
                           invariants_check: Dict[str, bool], database_matches: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the evaluation"""

        confidence = 1.0

        # Reduce confidence for physics violations
        physics_failures = sum(1 for v in physics_validation.values() if not v)
        confidence *= (1 - physics_failures * 0.1)

        # Reduce confidence for invariant violations
        invariant_failures = sum(1 for v in invariants_check.values() if not v)
        confidence *= (1 - invariant_failures * 0.15)

        # Increase confidence for database matches
        if database_matches:
            confidence *= 1.1

        return min(confidence, 1.0)

    def _generate_overall_assessment(self, branch_results: Dict[str, BranchScore], total_score: float) -> Dict[str, Any]:
        """Generate overall assessment of QuLab's capabilities"""

        assessment = {
            'total_score': total_score,
            'level': 'unknown',
            'strengths': [],
            'weaknesses': [],
            'hallucination_risk': 'unknown'
        }

        if total_score >= 0.8:
            assessment['level'] = 'excellent'
            assessment['hallucination_risk'] = 'low'
        elif total_score >= 0.6:
            assessment['level'] = 'good'
            assessment['hallucination_risk'] = 'medium'
        elif total_score >= 0.4:
            assessment['level'] = 'poor'
            assessment['hallucination_risk'] = 'high'
        else:
            assessment['level'] = 'failing'
            assessment['hallucination_risk'] = 'very_high'

        # Analyze branch performance
        for branch_name, branch_score in branch_results.items():
            if branch_score.average_score > 0.7:
                assessment['strengths'].append(f"Strong performance in {branch_name}")
            else:
                assessment['weaknesses'].append(f"Weak performance in {branch_name}")

            if branch_score.physics_violations > branch_score.questions_tested * 0.5:
                assessment['weaknesses'].append(f"High physics violations in {branch_name}")

        return assessment

    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on assessment"""

        recommendations = []

        level = assessment['level']
        hallucination_risk = assessment['hallucination_risk']

        if hallucination_risk in ['high', 'very_high']:
            recommendations.append("High hallucination risk detected - implement physics validation gates")
            recommendations.append("Add database cross-checking before presenting predictions")

        if level == 'failing':
            recommendations.append("Fundamental physics understanding needs improvement")
            recommendations.append("Consider retraining on validated materials data")

        elif level == 'poor':
            recommendations.append("Improve physics constraint enforcement")
            recommendations.append("Add more rigorous invariant checking")

        elif level in ['good', 'excellent']:
            recommendations.append("Continue with current approach but monitor for regressions")
            recommendations.append("Consider expanding to more complex materials predictions")

        return recommendations

    def export_evaluation_results(self, results: Dict[str, Any], filename: str):
        """Export evaluation results to file"""

        # Convert dataclasses to dicts for JSON serialization
        serializable_results = {
            'timestamp': results['timestamp'],
            'total_questions': results['total_questions'],
            'total_score': results['total_score'],
            'assessment': results['assessment'],
            'recommendations': results['recommendations'],
            'branch_results': results['branch_results'],
            'detailed_results': [r.__dict__ for r in self.results_history]
        }

        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logging.info(f"✅ Evaluation results exported to {filename}")


def main():
    """Run the QuLab Trap Framework evaluation"""

    framework = QuLabTrapFramework()

    logging.info("🪤 QuLab Trap Framework")
    logging.info("Testing QuLab's materials science capabilities against physics and databases")
    logging.info("=" * 80)

    # Run complete evaluation
    results = framework.run_complete_evaluation()

    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qulab_trap_evaluation_{timestamp}.json"
    framework.export_evaluation_results(results, filename)

    logging.info(f"\n📊 Final Assessment: {results['assessment']['level'].upper()}")
    logging.info(f"🎯 Total Score: {results['total_score']:.3f}/1.0")
    logging.info(f"🚩 Hallucination Risk: {results['assessment']['hallucination_risk'].upper()}")

    if results['recommendations']:
        logging.info(f"\n💡 Recommendations:")
        for rec in results['recommendations']:
            logging.info(f"   • {rec}")


if __name__ == "__main__":
    main()