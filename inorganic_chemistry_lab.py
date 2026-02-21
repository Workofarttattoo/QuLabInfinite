"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

INORGANIC CHEMISTRY LAB
Production-ready inorganic chemistry algorithms and simulations.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import math
from scipy.constants import physical_constants, e, k, h, c, Avogadro, R
from scipy.optimize import minimize
from scipy.special import sph_harm

class CrystalSystem(Enum):
    """Seven crystal systems"""
    CUBIC = "cubic"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"

class LigandFieldStrength(Enum):
    """Spectrochemical series ordering"""
    WEAK_FIELD = ["I-", "Br-", "SCN-", "Cl-", "F-", "OH-", "H2O"]
    MEDIUM_FIELD = ["NH3", "en", "py", "bpy"]
    STRONG_FIELD = ["NO2-", "phen", "CN-", "CO"]

@dataclass
class TransitionMetal:
    """Transition metal ion representation"""
    symbol: str
    oxidation_state: int
    d_electrons: int
    spin_state: str  # "high" or "low"
    coordination_number: int = 6

    def get_electron_config(self) -> str:
        """Get d-orbital electron configuration"""
        if self.coordination_number == 6:  # Octahedral
            if self.spin_state == "high":
                # Fill all t2g and eg with parallel spins first
                t2g = min(self.d_electrons, 3)
                eg = max(0, min(self.d_electrons - 3, 2))
                remaining = max(0, self.d_electrons - 5)
                return f"t2g^{t2g+remaining}eg^{eg}"
            else:  # Low spin
                t2g = min(self.d_electrons, 6)
                eg = max(0, self.d_electrons - 6)
                return f"t2g^{t2g}eg^{eg}"
        elif self.coordination_number == 4:  # Tetrahedral
            e = min(self.d_electrons, 4)
            t2 = max(0, self.d_electrons - 4)
            return f"e^{e}t2^{t2}"
        return f"d^{self.d_electrons}"

@dataclass
class Ligand:
    """Ligand representation"""
    formula: str
    charge: int
    denticity: int  # Number of donor atoms
    field_strength: float  # Position in spectrochemical series
    donor_atoms: List[str]
    pi_acceptor: bool = False
    pi_donor: bool = False

    def get_cone_angle(self) -> float:
        """Calculate Tolman cone angle for ligand"""
        # Simplified cone angle calculation
        cone_angles = {
            "H2O": 104, "NH3": 107, "Cl-": 102, "Br-": 105,
            "I-": 107, "CN-": 105, "CO": 95, "PPh3": 145,
            "PMe3": 118, "en": 130, "bpy": 140, "phen": 145
        }
        return cone_angles.get(self.formula, 100)

@dataclass
class CoordinationComplex:
    """Coordination complex representation"""
    metal: TransitionMetal
    ligands: List[Ligand]
    geometry: str  # octahedral, tetrahedral, square_planar, etc.

    def get_total_charge(self) -> int:
        """Calculate total complex charge"""
        metal_charge = self.metal.oxidation_state
        ligand_charge = sum(lig.charge for lig in self.ligands)
        return metal_charge + ligand_charge

    def get_formula(self) -> str:
        """Generate complex formula"""
        ligand_counts = {}
        for lig in self.ligands:
            ligand_counts[lig.formula] = ligand_counts.get(lig.formula, 0) + 1

        formula = f"[{self.metal.symbol}"
        for lig_formula, count in ligand_counts.items():
            if count > 1:
                formula += f"({lig_formula}){count}"
            else:
                formula += lig_formula
        formula += f"]{self.get_total_charge():+d}" if self.get_total_charge() != 0 else "]"
        return formula

class InorganicChemistryLab:
    """Production-ready inorganic chemistry laboratory"""

    def __init__(self, temperature: float = 298.15):
        self.temperature = temperature
        self.R = R  # Gas constant

        # Crystal field splitting parameters (in cm^-1)
        self.delta_oct = {  # Octahedral field splitting
            "Cr3+": 17400, "Mn2+": 7800, "Fe2+": 10400, "Fe3+": 13700,
            "Co2+": 9300, "Co3+": 18200, "Ni2+": 8500, "Cu2+": 12600
        }

        # Racah parameters for inter-electronic repulsion
        self.racah_b = {
            "Cr3+": 918, "Mn2+": 960, "Fe2+": 1058, "Fe3+": 1015,
            "Co2+": 1120, "Co3+": 1065, "Ni2+": 1080, "Cu2+": 1240
        }

        # Standard reduction potentials (V vs SHE)
        self.standard_potentials = {
            "Fe3+/Fe2+": 0.771, "Cu2+/Cu": 0.337, "Ag+/Ag": 0.799,
            "Au3+/Au": 1.498, "Pt2+/Pt": 1.188, "Pd2+/Pd": 0.915,
            "Ni2+/Ni": -0.257, "Co2+/Co": -0.280, "Zn2+/Zn": -0.763,
            "Mn2+/Mn": -1.185, "Cr3+/Cr2+": -0.424, "V3+/V2+": -0.255
        }

    def calculate_crystal_field_splitting(self, metal: TransitionMetal,
                                         ligands: List[Ligand],
                                         geometry: str) -> Dict:
        """Calculate crystal field splitting energy and diagram"""
        result = {
            'delta': 0.0,  # Splitting energy in cm^-1
            'orbital_energies': {},
            'CFSE': 0.0,  # Crystal field stabilization energy
            'spin_state': metal.spin_state,
            'pairing_energy': 0.0
        }

        # Get base splitting for metal
        metal_key = f"{metal.symbol}{metal.oxidation_state}+"
        delta_base = self.delta_oct.get(metal_key, 10000)

        # Adjust for ligand field strength
        avg_field_strength = np.mean([lig.field_strength for lig in ligands])
        delta = delta_base * avg_field_strength

        if geometry == "octahedral":
            result['delta'] = delta
            # t2g is stabilized by -0.4Δo, eg is destabilized by 0.6Δo
            result['orbital_energies'] = {
                't2g': -0.4 * delta,
                'eg': 0.6 * delta
            }

            # Calculate CFSE based on electron configuration
            if metal.spin_state == "high":
                t2g_e = min(metal.d_electrons, 3)
                eg_e = max(0, min(metal.d_electrons - 3, 2))
                remaining_t2g = max(0, metal.d_electrons - 5)
            else:  # Low spin
                t2g_e = min(metal.d_electrons, 6)
                eg_e = max(0, metal.d_electrons - 6)
                remaining_t2g = 0

            result['CFSE'] = (-0.4 * (t2g_e + remaining_t2g) + 0.6 * eg_e) * delta

        elif geometry == "tetrahedral":
            # Tetrahedral splitting is ~4/9 of octahedral
            delta_tet = delta * 4/9
            result['delta'] = delta_tet
            result['orbital_energies'] = {
                'e': -0.6 * delta_tet,
                't2': 0.4 * delta_tet
            }

            e_e = min(metal.d_electrons, 4)
            t2_e = max(0, metal.d_electrons - 4)
            result['CFSE'] = (-0.6 * e_e + 0.4 * t2_e) * delta_tet

        elif geometry == "square_planar":
            # Square planar has large splitting
            result['delta'] = delta * 1.3
            result['orbital_energies'] = {
                'dx2-y2': 1.0 * delta,
                'dxy': 0.5 * delta,
                'dz2': -0.4 * delta,
                'dxz,dyz': -0.6 * delta
            }

        # Calculate pairing energy
        if metal.spin_state == "low":
            racah_b = self.racah_b.get(metal_key, 1000)
            n_paired = max(0, metal.d_electrons - 5)
            result['pairing_energy'] = n_paired * racah_b

        return result

    def calculate_ligand_field_parameters(self, complex: CoordinationComplex) -> Dict:
        """Calculate comprehensive ligand field parameters"""
        params = {
            'dq': 0.0,  # Crystal field strength parameter
            'ds': 0.0,  # Tetragonal distortion parameter
            'dt': 0.0,  # Trigonal distortion parameter
            'cp': 0.0,  # Nephelauxetic ratio
            'B': 0.0,   # Racah parameter
            'C': 0.0,   # Racah parameter
            '10Dq': 0.0 # Crystal field splitting
        }

        # Calculate 10Dq (crystal field splitting)
        metal_key = f"{complex.metal.symbol}{complex.metal.oxidation_state}+"
        base_10dq = self.delta_oct.get(metal_key, 10000)

        # Apply ligand contributions
        f_values = []
        g_values = []
        for ligand in complex.ligands:
            # f and g values from spectrochemical series
            if ligand.formula in ["I-", "Br-", "Cl-"]:
                f_values.append(0.7 + 0.1 * ligand.field_strength)
                g_values.append(0.8)
            elif ligand.formula in ["H2O", "OH-"]:
                f_values.append(1.0)
                g_values.append(1.0)
            elif ligand.formula in ["NH3", "en"]:
                f_values.append(1.25)
                g_values.append(1.2)
            elif ligand.formula in ["CN-", "CO"]:
                f_values.append(1.7)
                g_values.append(1.5)
            else:
                f_values.append(1.0)
                g_values.append(1.0)

        f_avg = np.mean(f_values)
        g_avg = np.mean(g_values)

        params['10Dq'] = base_10dq * f_avg
        params['dq'] = params['10Dq'] / 10

        # Racah parameters
        B_free = self.racah_b.get(metal_key, 1000)
        params['B'] = B_free * g_avg * 0.85  # Nephelauxetic reduction
        params['C'] = 4 * params['B']  # Approximation C/B ≈ 4
        params['cp'] = params['B'] / B_free  # Nephelauxetic ratio

        # Distortion parameters (simplified)
        if complex.geometry != "octahedral":
            params['ds'] = 0.1 * params['10Dq']  # Tetragonal distortion
            params['dt'] = 0.05 * params['10Dq']  # Trigonal distortion

        return params

    def predict_complex_stability(self, complex: CoordinationComplex) -> Dict:
        """Predict stability constants and formation energies"""
        stability = {
            'log_beta': 0.0,  # Overall stability constant
            'stepwise_K': [],  # Stepwise formation constants
            'delta_G': 0.0,   # Gibbs free energy of formation
            'irving_williams_order': 0,  # Position in Irving-Williams series
            'chelate_effect': 0.0,  # Stability enhancement from chelation
            'hsab_compatibility': 0.0  # Hard-soft acid-base compatibility
        }

        # Irving-Williams series order
        irving_williams = ["Mn2+", "Fe2+", "Co2+", "Ni2+", "Cu2+", "Zn2+"]
        metal_key = f"{complex.metal.symbol}{complex.metal.oxidation_state}+"
        if metal_key in irving_williams:
            stability['irving_williams_order'] = irving_williams.index(metal_key) + 1
            # Base stability increases along the series
            stability['log_beta'] = 5 + stability['irving_williams_order'] * 2
        else:
            stability['log_beta'] = 10  # Default value

        # Apply ligand effects
        for ligand in complex.ligands:
            # Chelate effect
            if ligand.denticity > 1:
                stability['chelate_effect'] += (ligand.denticity - 1) * 2.3
                stability['log_beta'] += stability['chelate_effect']

            # HSAB principle
            # Hard acids: small, high charge density (Cr3+, Fe3+, Co3+)
            # Soft acids: large, polarizable (Cu+, Ag+, Au+, Hg2+)
            metal_hardness = self._get_metal_hardness(complex.metal)
            ligand_hardness = self._get_ligand_hardness(ligand)

            # Similar hardness = more stable
            hsab_match = 1 - abs(metal_hardness - ligand_hardness)
            stability['hsab_compatibility'] += hsab_match
            stability['log_beta'] += hsab_match * 1.5

        # Calculate Gibbs free energy
        stability['delta_G'] = -self.R * self.temperature * np.log(10) * stability['log_beta'] / 1000  # kJ/mol

        # Estimate stepwise formation constants
        n_ligands = len(set(lig.formula for lig in complex.ligands))
        for i in range(n_ligands):
            # Stepwise constants typically decrease
            K_step = 10 ** (stability['log_beta'] / n_ligands * (1 - i * 0.2))
            stability['stepwise_K'].append(K_step)

        return stability

    def _get_metal_hardness(self, metal: TransitionMetal) -> float:
        """Get hardness parameter for metal (0=soft, 1=hard)"""
        hardness_map = {
            ("Cr", 3): 0.9, ("Mn", 2): 0.8, ("Fe", 2): 0.7, ("Fe", 3): 0.9,
            ("Co", 2): 0.7, ("Co", 3): 0.8, ("Ni", 2): 0.6, ("Cu", 1): 0.2,
            ("Cu", 2): 0.5, ("Zn", 2): 0.7, ("Ag", 1): 0.1, ("Au", 1): 0.0,
            ("Au", 3): 0.3, ("Hg", 2): 0.1, ("Pd", 2): 0.3, ("Pt", 2): 0.3
        }
        return hardness_map.get((metal.symbol, metal.oxidation_state), 0.5)

    def _get_ligand_hardness(self, ligand: Ligand) -> float:
        """Get hardness parameter for ligand (0=soft, 1=hard)"""
        hardness_map = {
            "F-": 1.0, "OH-": 0.9, "H2O": 0.9, "NH3": 0.8,
            "Cl-": 0.6, "Br-": 0.4, "I-": 0.2, "CN-": 0.3,
            "CO": 0.2, "SCN-": 0.3, "S2-": 0.1, "PPh3": 0.1,
            "en": 0.7, "bpy": 0.5, "phen": 0.4
        }
        return hardness_map.get(ligand.formula, 0.5)

    def calculate_metal_ligand_bonding(self, complex: CoordinationComplex) -> Dict:
        """Analyze metal-ligand bonding using MO theory"""
        bonding = {
            'sigma_bonds': 0,
            'pi_bonds': 0,
            'delta_bonds': 0,
            'bond_order': {},
            'overlap_integrals': {},
            'covalent_character': 0.0,
            'ionic_character': 0.0
        }

        # Count sigma bonds (one per ligand coordination)
        for ligand in complex.ligands:
            bonding['sigma_bonds'] += ligand.denticity

        # Analyze pi bonding
        for ligand in complex.ligands:
            if ligand.pi_acceptor:  # CO, CN-, etc.
                bonding['pi_bonds'] += 1
                # Back-bonding from filled metal d orbitals to ligand pi*
                bonding['bond_order'][ligand.formula] = 1.5
            elif ligand.pi_donor:  # Cl-, O2-, etc.
                bonding['pi_bonds'] += 0.5
                bonding['bond_order'][ligand.formula] = 1.25
            else:
                bonding['bond_order'][ligand.formula] = 1.0

        # Calculate overlap integrals (simplified)
        for ligand in complex.ligands:
            # Overlap depends on orbital size match and symmetry
            metal_radius = self._get_metal_radius(complex.metal)
            ligand_radius = self._get_ligand_radius(ligand)

            # S = exp(-|R1-R2|/2) approximation
            overlap = np.exp(-abs(metal_radius - ligand_radius) / 2)
            bonding['overlap_integrals'][ligand.formula] = overlap

        # Estimate covalent vs ionic character
        electronegativity_diff = []
        for ligand in complex.ligands:
            # Simplified electronegativity difference
            metal_en = 1.5 + complex.metal.oxidation_state * 0.2
            ligand_en = self._get_ligand_electronegativity(ligand)
            electronegativity_diff.append(abs(metal_en - ligand_en))

        avg_en_diff = np.mean(electronegativity_diff)
        bonding['ionic_character'] = 1 - np.exp(-0.25 * avg_en_diff ** 2)
        bonding['covalent_character'] = 1 - bonding['ionic_character']

        return bonding

    def _get_metal_radius(self, metal: TransitionMetal) -> float:
        """Get ionic radius in Angstroms"""
        radius_map = {
            ("Cr", 3): 0.615, ("Mn", 2): 0.83, ("Fe", 2): 0.78, ("Fe", 3): 0.645,
            ("Co", 2): 0.745, ("Co", 3): 0.61, ("Ni", 2): 0.69, ("Cu", 2): 0.73,
            ("Zn", 2): 0.74, ("Ag", 1): 1.15, ("Au", 3): 0.85
        }
        return radius_map.get((metal.symbol, metal.oxidation_state), 0.7)

    def _get_ligand_radius(self, ligand: Ligand) -> float:
        """Get ligand donor atom radius"""
        radius_map = {
            "F-": 1.33, "Cl-": 1.81, "Br-": 1.96, "I-": 2.20,
            "O": 1.40, "N": 1.55, "S": 1.80, "P": 1.80, "C": 1.70
        }
        if ligand.donor_atoms:
            return radius_map.get(ligand.donor_atoms[0], 1.5)
        return 1.5

    def _get_ligand_electronegativity(self, ligand: Ligand) -> float:
        """Get ligand electronegativity"""
        en_map = {
            "F-": 4.0, "Cl-": 3.0, "Br-": 2.8, "I-": 2.5,
            "OH-": 3.5, "H2O": 3.5, "NH3": 3.0, "CN-": 2.5,
            "CO": 2.5, "PPh3": 2.1
        }
        return en_map.get(ligand.formula, 2.5)

    def calculate_redox_potential(self, half_cell1: str, half_cell2: str,
                                 concentrations: Dict[str, float] = None) -> Dict:
        """Calculate cell potential using Nernst equation"""
        result = {
            'E_cell_standard': 0.0,  # Standard cell potential (V)
            'E_cell': 0.0,           # Actual cell potential (V)
            'delta_G': 0.0,          # Gibbs free energy (kJ/mol)
            'K_eq': 0.0,             # Equilibrium constant
            'spontaneous': False,
            'electrons_transferred': 0
        }

        # Get standard potentials
        E1 = self.standard_potentials.get(half_cell1, 0.0)
        E2 = self.standard_potentials.get(half_cell2, 0.0)

        # Determine cathode (reduction) and anode (oxidation)
        if E1 > E2:
            E_cathode = E1
            E_anode = E2
            cathode = half_cell1
            anode = half_cell2
        else:
            E_cathode = E2
            E_anode = E1
            cathode = half_cell2
            anode = half_cell1

        result['E_cell_standard'] = E_cathode - E_anode

        # Determine number of electrons (simplified)
        n = 2  # Default to 2 electron transfer
        if "3+" in cathode or "3+" in anode:
            n = 3
        elif "1+" in cathode or "1+" in anode:
            n = 1
        result['electrons_transferred'] = n

        # Apply Nernst equation if concentrations provided
        if concentrations:
            # E = E° - (RT/nF)lnQ
            Q = 1.0  # Reaction quotient
            # Simplified: assume 1:1 stoichiometry
            if cathode.split('/')[0] in concentrations and anode.split('/')[0] in concentrations:
                Q = concentrations[anode.split('/')[0]] / concentrations[cathode.split('/')[0]]

            nernst_term = (self.R * self.temperature) / (n * 96485) * np.log(Q)
            result['E_cell'] = result['E_cell_standard'] - nernst_term
        else:
            result['E_cell'] = result['E_cell_standard']

        # Calculate thermodynamics
        result['delta_G'] = -n * 96485 * result['E_cell'] / 1000  # kJ/mol
        result['K_eq'] = np.exp(-result['delta_G'] * 1000 / (self.R * self.temperature))
        result['spontaneous'] = result['E_cell'] > 0

        return result

    def analyze_spectrochemical_series(self, ligands: List[Ligand]) -> Dict:
        """Analyze ligands position in spectrochemical series"""
        analysis = {
            'ordering': [],
            'field_strengths': {},
            'pi_effects': {},
            'predicted_delta': 0.0,
            'spin_crossover_likelihood': 0.0
        }

        # Standard spectrochemical series
        series_order = {
            "I-": 1, "Br-": 2, "SCN-": 3, "Cl-": 4, "S2-": 5,
            "F-": 6, "OH-": 7, "ox2-": 8, "H2O": 9, "NCS-": 10,
            "NH3": 11, "en": 12, "bpy": 13, "phen": 14, "NO2-": 15,
            "PPh3": 16, "CN-": 17, "CO": 18
        }

        # Sort ligands by field strength
        sorted_ligands = sorted(ligands,
                               key=lambda l: series_order.get(l.formula, 10))

        for ligand in sorted_ligands:
            position = series_order.get(ligand.formula, 10)
            analysis['ordering'].append((ligand.formula, position))
            analysis['field_strengths'][ligand.formula] = ligand.field_strength

            # Analyze pi effects
            if ligand.pi_acceptor:
                analysis['pi_effects'][ligand.formula] = "pi-acceptor (increases Δ)"
            elif ligand.pi_donor:
                analysis['pi_effects'][ligand.formula] = "pi-donor (decreases Δ)"
            else:
                analysis['pi_effects'][ligand.formula] = "sigma-only"

        # Predict crystal field splitting
        avg_position = np.mean([series_order.get(l.formula, 10) for l in ligands])
        analysis['predicted_delta'] = 5000 + avg_position * 1000  # cm^-1

        # Assess spin crossover likelihood
        # Intermediate field strength ligands more likely to show spin crossover
        if 8 < avg_position < 14:
            analysis['spin_crossover_likelihood'] = 0.7
        else:
            analysis['spin_crossover_likelihood'] = 0.2

        return analysis

    def calculate_tanabe_sugano_parameters(self, metal: TransitionMetal,
                                          field_strength: float) -> Dict:
        """Calculate Tanabe-Sugano diagram parameters for d-electron systems"""
        params = {
            'ground_state': "",
            'excited_states': [],
            'transition_energies': {},
            'B_value': 0.0,
            'C_value': 0.0,
            'delta_over_B': 0.0
        }

        metal_key = f"{metal.symbol}{metal.oxidation_state}+"
        B = self.racah_b.get(metal_key, 1000)
        C = 4 * B

        params['B_value'] = B
        params['C_value'] = C
        params['delta_over_B'] = field_strength / B

        # Determine ground state term symbol
        if metal.d_electrons == 3:  # d3
            params['ground_state'] = "4A2g"
            params['excited_states'] = ["4T2g", "4T1g(F)", "4T1g(P)"]
            # Transition energies (simplified)
            params['transition_energies'] = {
                "4A2g→4T2g": field_strength,
                "4A2g→4T1g(F)": 1.5 * field_strength,
                "4A2g→4T1g(P)": 2.2 * field_strength
            }
        elif metal.d_electrons == 6:  # d6
            if metal.spin_state == "high":
                params['ground_state'] = "5T2g"
                params['excited_states'] = ["5Eg"]
                params['transition_energies'] = {
                    "5T2g→5Eg": field_strength
                }
            else:  # Low spin
                params['ground_state'] = "1A1g"
                params['excited_states'] = ["1T1g", "1T2g"]
                params['transition_energies'] = {
                    "1A1g→1T1g": field_strength,
                    "1A1g→1T2g": 1.6 * field_strength
                }
        elif metal.d_electrons == 8:  # d8
            params['ground_state'] = "3A2g"
            params['excited_states'] = ["3T2g", "3T1g(F)", "3T1g(P)"]
            params['transition_energies'] = {
                "3A2g→3T2g": field_strength,
                "3A2g→3T1g(F)": 1.5 * field_strength,
                "3A2g→3T1g(P)": 2.5 * field_strength
            }

        return params

    def predict_magnetic_properties(self, complex: CoordinationComplex) -> Dict:
        """Predict magnetic moment and susceptibility"""
        magnetic = {
            'n_unpaired': 0,
            'spin_only_moment': 0.0,  # Bohr magnetons
            'orbital_contribution': 0.0,
            'effective_moment': 0.0,
            'susceptibility': 0.0,  # cm3/mol
            'magnetic_type': ""
        }

        # Determine number of unpaired electrons
        d_electrons = complex.metal.d_electrons

        if complex.geometry == "octahedral":
            if complex.metal.spin_state == "high":
                # High spin fills orbitals singly first
                if d_electrons <= 5:
                    magnetic['n_unpaired'] = d_electrons
                else:
                    magnetic['n_unpaired'] = 10 - d_electrons
            else:  # Low spin
                if d_electrons <= 3:
                    magnetic['n_unpaired'] = d_electrons
                elif d_electrons <= 6:
                    magnetic['n_unpaired'] = 6 - d_electrons
                else:
                    magnetic['n_unpaired'] = d_electrons - 6

        elif complex.geometry == "tetrahedral":
            # Tetrahedral is usually high spin
            if d_electrons <= 4:
                magnetic['n_unpaired'] = d_electrons
            else:
                magnetic['n_unpaired'] = 10 - d_electrons

        # Calculate spin-only magnetic moment
        n = magnetic['n_unpaired']
        magnetic['spin_only_moment'] = np.sqrt(n * (n + 2))

        # Estimate orbital contribution
        # First row transition metals have quenched orbital momentum
        if complex.metal.symbol in ["Fe", "Co", "Ni"]:
            magnetic['orbital_contribution'] = 0.1 * magnetic['spin_only_moment']

        magnetic['effective_moment'] = magnetic['spin_only_moment'] + magnetic['orbital_contribution']

        # Calculate molar susceptibility (CGS units)
        if magnetic['n_unpaired'] > 0:
            magnetic['susceptibility'] = (magnetic['effective_moment'] ** 2 * 0.125) / self.temperature
            magnetic['magnetic_type'] = "paramagnetic"
        else:
            magnetic['susceptibility'] = -1e-6  # Diamagnetic correction
            magnetic['magnetic_type'] = "diamagnetic"

        return magnetic

    def calculate_lattice_energy(self, cation: Dict, anion: Dict,
                                structure_type: str) -> Dict:
        """Calculate lattice energy using Born-Landé equation"""
        result = {
            'lattice_energy': 0.0,  # kJ/mol
            'madelung_constant': 0.0,
            'born_exponent': 0.0,
            'interionic_distance': 0.0,  # Angstroms
            'lattice_enthalpy': 0.0
        }

        # Madelung constants for common structures
        madelung = {
            "rock_salt": 1.7476, "cesium_chloride": 1.7627,
            "fluorite": 2.5194, "rutile": 2.4080,
            "zinc_blende": 1.6381, "wurtzite": 1.641
        }

        result['madelung_constant'] = madelung.get(structure_type, 1.7476)

        # Calculate interionic distance (sum of ionic radii)
        r_cation = cation.get('radius', 1.0)  # Angstroms
        r_anion = anion.get('radius', 1.5)
        result['interionic_distance'] = r_cation + r_anion

        # Born exponent based on electron configuration
        n_values = {
            "He": 5, "Ne": 7, "Ar": 9, "Kr": 10, "Xe": 12
        }
        # Determine from noble gas configuration
        result['born_exponent'] = 9  # Default for Ar configuration

        # Born-Landé equation
        # U = -M * z+ * z- * e^2 * N_A / (4πε₀r₀) * (1 - 1/n)
        z_plus = cation.get('charge', 1)
        z_minus = abs(anion.get('charge', -1))
        M = result['madelung_constant']
        r0 = result['interionic_distance'] * 1e-10  # Convert to meters
        n = result['born_exponent']

        # Constants
        k_e = 8.99e9  # Coulomb constant
        N_A = Avogadro
        e_charge = e

        U = M * z_plus * z_minus * e_charge**2 * k_e * N_A / r0 * (1 - 1/n)
        result['lattice_energy'] = U / 1000  # Convert to kJ/mol

        # Lattice enthalpy (slight correction for PV work)
        result['lattice_enthalpy'] = result['lattice_energy'] - 2.5 * self.R * self.temperature / 1000

        return result

def run_demo():
    """Demonstrate inorganic chemistry lab capabilities"""
    print("=" * 80)
    print("INORGANIC CHEMISTRY LAB - Production Demo")
    print("Copyright (c) 2025 Joshua Hendricks Cole")
    print("=" * 80)

    # Initialize lab
    lab = InorganicChemistryLab(temperature=298.15)

    # Create a coordination complex
    print("\n1. Creating coordination complex...")

    # Create Fe(III) complex with mixed ligands
    metal = TransitionMetal("Fe", oxidation_state=3, d_electrons=5,
                           spin_state="high", coordination_number=6)

    ligands = [
        Ligand("H2O", charge=0, denticity=1, field_strength=0.9,
               donor_atoms=["O"]),
        Ligand("H2O", charge=0, denticity=1, field_strength=0.9,
               donor_atoms=["O"]),
        Ligand("Cl-", charge=-1, denticity=1, field_strength=0.4,
               donor_atoms=["Cl"]),
        Ligand("Cl-", charge=-1, denticity=1, field_strength=0.4,
               donor_atoms=["Cl"]),
        Ligand("NH3", charge=0, denticity=1, field_strength=1.25,
               donor_atoms=["N"]),
        Ligand("NH3", charge=0, denticity=1, field_strength=1.25,
               donor_atoms=["N"])
    ]

    complex = CoordinationComplex(metal, ligands, "octahedral")
    print(f"   Complex formula: {complex.get_formula()}")
    print(f"   Electron configuration: {metal.get_electron_config()}")

    # Crystal field theory
    print("\n2. Crystal field splitting analysis...")
    cf_result = lab.calculate_crystal_field_splitting(metal, ligands, "octahedral")
    print(f"   Crystal field splitting (Δo): {cf_result['delta']:.0f} cm⁻¹")
    print(f"   CFSE: {cf_result['CFSE']:.0f} cm⁻¹")
    print(f"   Spin state: {cf_result['spin_state']}")

    # Ligand field parameters
    print("\n3. Ligand field parameters...")
    lf_params = lab.calculate_ligand_field_parameters(complex)
    print(f"   10Dq: {lf_params['10Dq']:.0f} cm⁻¹")
    print(f"   Racah B: {lf_params['B']:.0f} cm⁻¹")
    print(f"   Nephelauxetic ratio: {lf_params['cp']:.3f}")

    # Complex stability
    print("\n4. Complex stability analysis...")
    stability = lab.predict_complex_stability(complex)
    print(f"   Log β (stability constant): {stability['log_beta']:.1f}")
    print(f"   ΔG formation: {stability['delta_G']:.1f} kJ/mol")
    print(f"   HSAB compatibility: {stability['hsab_compatibility']:.2f}")

    # Metal-ligand bonding
    print("\n5. Metal-ligand bonding analysis...")
    bonding = lab.calculate_metal_ligand_bonding(complex)
    print(f"   Sigma bonds: {bonding['sigma_bonds']}")
    print(f"   Pi bonds: {bonding['pi_bonds']}")
    print(f"   Covalent character: {bonding['covalent_character']:.1%}")
    print(f"   Ionic character: {bonding['ionic_character']:.1%}")

    # Redox potential
    print("\n6. Redox potential calculation...")
    redox = lab.calculate_redox_potential("Fe3+/Fe2+", "Cu2+/Cu",
                                         {"Fe3+": 0.01, "Cu2+": 0.1})
    print(f"   E°cell: {redox['E_cell_standard']:.3f} V")
    print(f"   Ecell: {redox['E_cell']:.3f} V")
    print(f"   ΔG: {redox['delta_G']:.1f} kJ/mol")
    print(f"   Spontaneous: {redox['spontaneous']}")

    # Spectrochemical series
    print("\n7. Spectrochemical series analysis...")
    spec_analysis = lab.analyze_spectrochemical_series(ligands[:3])
    print(f"   Ligand ordering: {spec_analysis['ordering']}")
    print(f"   Predicted Δ: {spec_analysis['predicted_delta']:.0f} cm⁻¹")

    # Tanabe-Sugano parameters
    print("\n8. Tanabe-Sugano diagram parameters...")
    ts_params = lab.calculate_tanabe_sugano_parameters(metal, cf_result['delta'])
    print(f"   Ground state: {ts_params['ground_state']}")
    print(f"   Δ/B ratio: {ts_params['delta_over_B']:.1f}")
    if ts_params['transition_energies']:
        for transition, energy in list(ts_params['transition_energies'].items())[:2]:
            print(f"   {transition}: {energy:.0f} cm⁻¹")

    # Magnetic properties
    print("\n9. Magnetic properties...")
    magnetic = lab.predict_magnetic_properties(complex)
    print(f"   Unpaired electrons: {magnetic['n_unpaired']}")
    print(f"   μeff: {magnetic['effective_moment']:.2f} BM")
    print(f"   Magnetic type: {magnetic['magnetic_type']}")

    # Lattice energy
    print("\n10. Lattice energy calculation (NaCl)...")
    cation = {'charge': 1, 'radius': 1.02}  # Na+
    anion = {'charge': -1, 'radius': 1.81}  # Cl-
    lattice = lab.calculate_lattice_energy(cation, anion, "rock_salt")
    print(f"   Lattice energy: {lattice['lattice_energy']:.0f} kJ/mol")
    print(f"   Madelung constant: {lattice['madelung_constant']:.4f}")
    print(f"   Interionic distance: {lattice['interionic_distance']:.2f} Å")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("Visit: https://aios.is | https://thegavl.com")
    print("=" * 80)

if __name__ == '__main__':
    run_demo()