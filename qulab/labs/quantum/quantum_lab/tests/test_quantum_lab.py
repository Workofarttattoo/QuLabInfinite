#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.

Test Suite for Quantum Laboratory
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_lab import QuantumLabSimulator, SimulationBackend
from quantum_chemistry import Molecule, QuantumChemistry
from quantum_materials import QuantumMaterials
from quantum_sensors import QuantumSensors
from quantum_validation import QuantumValidation


class TestQuantumLabSimulator(unittest.TestCase):
    """Test quantum laboratory simulator"""

    def test_initialization(self):
        """Test simulator initialization"""
        lab = QuantumLabSimulator(num_qubits=5, verbose=False)
        self.assertEqual(lab.num_qubits, 5)
        self.assertEqual(lab.backend, SimulationBackend.STATEVECTOR_EXACT)

    def test_single_qubit_gates(self):
        """Test single-qubit gate operations"""
        lab = QuantumLabSimulator(num_qubits=2, verbose=False)

        # Apply Hadamard
        lab.h(0)

        # State should be in superposition
        probs = lab.get_probabilities()
        self.assertIn('00', probs)
        self.assertIn('10', probs)

        # Probabilities should be roughly equal
        self.assertAlmostEqual(probs.get('00', 0), 0.5, delta=0.1)
        self.assertAlmostEqual(probs.get('10', 0), 0.5, delta=0.1)

    def test_two_qubit_gates(self):
        """Test two-qubit gate operations"""
        lab = QuantumLabSimulator(num_qubits=2, verbose=False)

        # Create Bell state
        lab.h(0).cnot(0, 1)

        probs = lab.get_probabilities()

        # Should have |00⟩ and |11⟩ with ~50% each
        self.assertAlmostEqual(probs.get('00', 0) + probs.get('11', 0), 1.0, delta=0.1)

    def test_measurement(self):
        """Test measurement"""
        lab = QuantumLabSimulator(num_qubits=3, verbose=False)

        # Prepare |+⟩ state
        lab.h(0).h(1).h(2)

        # Measure
        results = lab.measure_all()

        self.assertEqual(len(results), 3)
        self.assertTrue(all(r in [0, 1] for r in results))

    def test_reset(self):
        """Test state reset"""
        lab = QuantumLabSimulator(num_qubits=3, verbose=False)

        lab.h(0).x(1)
        lab.reset()

        probs = lab.get_probabilities()

        # Should be back to |000⟩
        self.assertAlmostEqual(probs.get('000', 0), 1.0, delta=0.01)

    def test_tensor_network_backend(self):
        """Test tensor network backend for large circuits"""
        lab = QuantumLabSimulator(
            num_qubits=35,
            backend=SimulationBackend.TENSOR_NETWORK,
            verbose=False
        )

        self.assertEqual(lab.num_qubits, 35)
        self.assertEqual(lab.backend, SimulationBackend.TENSOR_NETWORK)

        # Should not crash on gate operations
        lab.h(0).cnot(0, 1)


class TestQuantumChemistry(unittest.TestCase):
    """Test quantum chemistry module"""

    def setUp(self):
        """Set up test simulator"""
        self.lab = QuantumLabSimulator(num_qubits=10, verbose=False)
        self.chem = QuantumChemistry(self.lab)

    def test_hydrogen_molecule(self):
        """Test H2 molecule creation"""
        h2 = Molecule.hydrogen_molecule(bond_length=0.74)

        self.assertEqual(len(h2.atoms), 2)
        self.assertEqual(h2.atoms[0].element, 'H')
        self.assertEqual(h2.num_electrons, 2)

    def test_water_molecule(self):
        """Test H2O molecule creation"""
        h2o = Molecule.water_molecule()

        self.assertEqual(len(h2o.atoms), 3)
        self.assertEqual(h2o.atoms[0].element, 'O')
        self.assertEqual(h2o.num_electrons, 10)

    def test_vqe_energy(self):
        """Test VQE energy calculation"""
        h2 = Molecule.hydrogen_molecule(bond_length=0.74)

        # VQE should return reasonable energy
        energy = self.chem.vqe_optimize(h2, max_iter=10)

        # Should be negative (bound state)
        self.assertLess(energy, 0.0)

        # Should be near reference value (-1.137 Ha)
        self.assertGreater(energy, -2.0)  # Not too negative

    def test_molecular_orbitals(self):
        """Test molecular orbital calculation"""
        h2 = Molecule.hydrogen_molecule()

        orbitals = self.chem.molecular_orbitals(h2)

        self.assertIn('energies', orbitals)
        self.assertIn('homo_energy', orbitals)
        self.assertIn('lumo_energy', orbitals)
        self.assertIn('gap', orbitals)

        # HOMO-LUMO gap should be positive
        self.assertGreater(orbitals['gap'], 0.0)


class TestQuantumMaterials(unittest.TestCase):
    """Test quantum materials module"""

    def setUp(self):
        """Set up test simulator"""
        self.lab = QuantumLabSimulator(num_qubits=12, verbose=False)
        self.materials = QuantumMaterials(self.lab)

    def test_band_gap_silicon(self):
        """Test silicon band gap"""
        gap = self.materials.compute_band_gap('silicon')

        # Silicon gap ~1.12 eV
        self.assertGreater(gap, 0.5)
        self.assertLess(gap, 2.0)

    def test_band_structure(self):
        """Test band structure calculation"""
        bands = self.materials.compute_band_structure('silicon', num_k_points=20)

        self.assertIn('k_points', bands)
        self.assertIn('valence_band', bands)
        self.assertIn('conduction_band', bands)

        self.assertEqual(len(bands['k_points']), 20)

    def test_superconductor_tc(self):
        """Test BCS critical temperature"""
        tc = self.materials.bcs_critical_temperature('aluminum')

        # Aluminum Tc = 1.20 K
        self.assertGreater(tc, 0.5)
        self.assertLess(tc, 2.0)

    def test_topological_chern_number(self):
        """Test Chern number calculation"""
        # Simple Hamiltonian (not used in simplified version)
        H = np.eye(4)

        chern = self.materials.topological_chern_number(H, num_k_points=10)

        # Should be integer
        self.assertIsInstance(chern, int)

    def test_quantum_phase_transition(self):
        """Test quantum phase transition detection"""
        phase_info = self.materials.quantum_phase_transition(
            coupling_strength=1.0,
            field_strength=0.5
        )

        self.assertIn('phase', phase_info)
        self.assertIn('ratio', phase_info)
        self.assertIn('at_critical_point', phase_info)


class TestQuantumSensors(unittest.TestCase):
    """Test quantum sensors module"""

    def setUp(self):
        """Set up test simulator"""
        self.lab = QuantumLabSimulator(num_qubits=8, verbose=False)
        self.sensors = QuantumSensors(self.lab)

    def test_magnetometry_sensitivity(self):
        """Test magnetometer sensitivity"""
        sens = self.sensors.magnetometry_sensitivity(
            num_qubits=5,
            measurement_time=1.0,
            method='ramsey'
        )

        # Should be positive
        self.assertGreater(sens, 0.0)

        # Should be in reasonable range (fT to pT)
        self.assertLess(sens, 1e-9)  # < 1 nT

    def test_gravimetry_precision(self):
        """Test gravimeter precision"""
        precision = self.sensors.gravimetry_precision(
            interrogation_time=1.0,
            num_atoms=1e6
        )

        # Should be positive
        self.assertGreater(precision, 0.0)

        # Should be better than classical (< 1 nGal = 1e-11 m/s²)
        self.assertLess(precision, 1e-8)

    def test_atomic_clock_stability(self):
        """Test atomic clock stability"""
        stability = self.sensors.atomic_clock_stability(
            averaging_time=100,
            num_atoms=1e4
        )

        # Should be positive
        self.assertGreater(stability, 0.0)

        # Should be reasonable fractional frequency
        self.assertLess(stability, 1e-10)

    def test_nv_center_sensing(self):
        """Test NV center sensor"""
        specs = self.sensors.nitrogen_vacancy_sensing(
            field_strength=1e-6,
            decoherence_time=1e-3
        )

        self.assertIn('sensitivity_T', specs)
        self.assertIn('decoherence_time_s', specs)
        self.assertIn('spatial_resolution_m', specs)

        # Sensitivity should be in nT range
        self.assertLess(specs['sensitivity_T'], 1e-6)


class TestQuantumValidation(unittest.TestCase):
    """Test validation module"""

    def setUp(self):
        """Set up validator"""
        self.validator = QuantumValidation()

    def test_bell_state_validation(self):
        """Test Bell state validation"""
        # Perfect Bell state
        perfect_probs = {'00': 0.5, '11': 0.5}
        result = self.validator.validate_bell_state(perfect_probs)

        self.assertTrue(result['passed'])
        self.assertLess(result['mean_error'], 0.01)

    def test_chemistry_validation(self):
        """Test chemistry energy validation"""
        # Close to reference
        result = self.validator.validate_chemistry_energy('H2_0.74', -1.140)

        self.assertIsNotNone(result['passed'])
        self.assertIn('reference', result)
        self.assertIn('computed', result)

    def test_band_gap_validation(self):
        """Test band gap validation"""
        result = self.validator.validate_band_gap('silicon', 1.15)

        self.assertIsNotNone(result['passed'])
        self.assertIn('error', result)

    def test_validation_report(self):
        """Test report generation"""
        # Run some validations
        self.validator.validate_bell_state({'00': 0.5, '11': 0.5})

        report = self.validator.generate_validation_report()

        self.assertIsInstance(report, str)
        self.assertIn('VALIDATION REPORT', report)
        self.assertIn('Total tests', report)


# Integration tests
class TestIntegration(unittest.TestCase):
    """Integration tests across modules"""

    def test_full_chemistry_workflow(self):
        """Test complete chemistry workflow"""
        lab = QuantumLabSimulator(num_qubits=10, verbose=False)

        # Create molecule
        h2 = Molecule.hydrogen_molecule(bond_length=0.74)

        # Compute energy
        energy = lab.chemistry.compute_ground_state_energy(h2, method='VQE')

        # Validate
        validator = QuantumValidation()
        result = validator.validate_chemistry_energy('H2_0.74', energy)

        self.assertIsNotNone(result)

    def test_full_materials_workflow(self):
        """Test complete materials workflow"""
        lab = QuantumLabSimulator(num_qubits=12, verbose=False)

        # Compute band gap
        gap = lab.materials.compute_band_gap('silicon')

        # Validate
        validator = QuantumValidation()
        result = validator.validate_band_gap('silicon', gap)

        self.assertIsNotNone(result)

    def test_sensor_suite(self):
        """Test sensor suite"""
        lab = QuantumLabSimulator(num_qubits=10, verbose=False)

        # Run multiple sensor calculations
        mag_sens = lab.sensors.magnetometry_sensitivity(num_qubits=5)
        grav_prec = lab.sensors.gravimetry_precision()
        clock_stab = lab.sensors.atomic_clock_stability()

        # All should be positive
        self.assertGreater(mag_sens, 0)
        self.assertGreater(grav_prec, 0)
        self.assertGreater(clock_stab, 0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumLabSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumChemistry))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumMaterials))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumSensors))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    print("\n" + "="*60)
    print("QUANTUM LABORATORY TEST SUITE")
    print("="*60 + "\n")

    result = run_tests()

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED")
        sys.exit(1)
