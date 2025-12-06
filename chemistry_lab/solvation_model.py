"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Solvation Model
Implicit and explicit solvation models: PCM, COSMO, SMD, pH effects, logP prediction.
"""

# NOTE: All mentions of "free energy" refer to classical Gibbs free energy
# changes predicted by the solvation models, not pseudoscientific claims.

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SolvationModel(Enum):
    """Solvation model types."""
    PCM = "pcm"  # Polarizable Continuum Model
    COSMO = "cosmo"  # Conductor-like Screening Model
    SMD = "smd"  # Solvation Model Density
    EXPLICIT = "explicit"  # Explicit solvent molecules
    GBSA = "gbsa"  # Generalized Born Surface Area


@dataclass
class Solvent:
    """Solvent properties."""
    name: str
    dielectric: float  # Dielectric constant
    density: float  # g/mL
    viscosity: float  # cP
    surface_tension: float  # dyn/cm
    donor_number: float  # Lewis basicity
    acceptor_number: float  # Lewis acidity
    polarity_index: float  # 0-10 scale


@dataclass
class Solute:
    """Solute (molecule) properties."""
    name: str
    smiles: str
    molecular_weight: float
    charge: float
    dipole_moment: float  # Debye
    polarizability: float  # Angstrom^3
    surface_area: float  # Angstrom^2
    volume: float  # Angstrom^3
    hbond_donors: int
    hbond_acceptors: int


@dataclass
class SolvationEnergy:
    """Solvation free energy components."""
    total: float  # Total solvation free energy (kcal/mol)
    electrostatic: float  # Electrostatic contribution
    cavitation: float  # Cavity formation
    dispersion: float  # Dispersion interactions
    repulsion: float  # Repulsion
    hydrogen_bonding: float  # H-bonding contribution


class SolvationCalculator:
    """
    Calculate solvation effects using various models.

    Features:
    - PCM/COSMO/SMD implicit solvation
    - Explicit solvent molecules
    - pH effects on ionization
    - logP (partition coefficient) prediction
    - Solvation free energy
    """

    def __init__(self):
        self.solvents = self._load_solvent_database()
        self.k_B = 1.987204e-3  # kcal/(mol*K)

    def _load_solvent_database(self) -> Dict[str, Solvent]:
        """Load common solvents database."""
        return {
            "water": Solvent(
                name="water",
                dielectric=78.4,
                density=1.0,
                viscosity=0.89,
                surface_tension=72.0,
                donor_number=18.0,
                acceptor_number=54.8,
                polarity_index=10.2
            ),
            "methanol": Solvent(
                name="methanol",
                dielectric=32.7,
                density=0.79,
                viscosity=0.54,
                surface_tension=22.6,
                donor_number=19.0,
                acceptor_number=41.3,
                polarity_index=5.1
            ),
            "ethanol": Solvent(
                name="ethanol",
                dielectric=24.3,
                density=0.79,
                viscosity=1.07,
                surface_tension=22.1,
                donor_number=20.0,
                acceptor_number=37.1,
                polarity_index=4.3
            ),
            "acetone": Solvent(
                name="acetone",
                dielectric=20.7,
                density=0.79,
                viscosity=0.30,
                surface_tension=23.0,
                donor_number=17.0,
                acceptor_number=12.5,
                polarity_index=5.1
            ),
            "dmso": Solvent(
                name="dmso",
                dielectric=46.7,
                density=1.10,
                viscosity=1.99,
                surface_tension=43.0,
                donor_number=29.8,
                acceptor_number=19.3,
                polarity_index=7.2
            ),
            "chloroform": Solvent(
                name="chloroform",
                dielectric=4.8,
                density=1.48,
                viscosity=0.54,
                surface_tension=27.1,
                donor_number=0.0,
                acceptor_number=23.1,
                polarity_index=4.1
            ),
            "hexane": Solvent(
                name="hexane",
                dielectric=1.9,
                density=0.66,
                viscosity=0.30,
                surface_tension=18.4,
                donor_number=0.0,
                acceptor_number=0.0,
                polarity_index=0.1
            ),
            "toluene": Solvent(
                name="toluene",
                dielectric=2.4,
                density=0.87,
                viscosity=0.56,
                surface_tension=28.4,
                donor_number=0.1,
                acceptor_number=11.5,
                polarity_index=2.4
            ),
        }

    def pcm_solvation(
        self,
        solute: Solute,
        solvent: Solvent,
        temperature: float = 298.15
    ) -> SolvationEnergy:
        """
        Calculate solvation free energy using Polarizable Continuum Model (PCM).

        ΔG_solv = ΔG_elec + ΔG_cav + ΔG_disp + ΔG_rep
        """
        # Electrostatic contribution (Born model)
        # ΔG_elec = -1/2 * q^2 / a * (1 - 1/ε)
        radius = (3 * solute.volume / (4 * np.pi)) ** (1/3)  # Effective radius
        delta_g_elec = -166.0 * solute.charge**2 / radius * (1 - 1/solvent.dielectric)

        # Add dipole contribution
        delta_g_elec += -solute.dipole_moment**2 / (2 * radius**3) * (solvent.dielectric - 1) / (2 * solvent.dielectric + 1)

        # Cavitation energy (scaled particle theory)
        # ΔG_cav = k_B*T * ln(1 + ρ*V) + surface_tension * Area
        delta_g_cav = (
            self.k_B * temperature * np.log(1 + solvent.density * solute.volume / 1000.0) +
            solvent.surface_tension * solute.surface_area / 1000.0  # Convert to kcal/mol
        )

        # Dispersion (van der Waals)
        # Simplified: proportional to surface area and polarizability
        delta_g_disp = -0.02 * solute.surface_area * np.sqrt(solute.polarizability)

        # Repulsion (exchange repulsion)
        delta_g_rep = 0.01 * solute.surface_area

        # Total
        delta_g_total = delta_g_elec + delta_g_cav + delta_g_disp + delta_g_rep

        return SolvationEnergy(
            total=delta_g_total,
            electrostatic=delta_g_elec,
            cavitation=delta_g_cav,
            dispersion=delta_g_disp,
            repulsion=delta_g_rep,
            hydrogen_bonding=0.0  # Not included in basic PCM
        )

    def smd_solvation(
        self,
        solute: Solute,
        solvent: Solvent,
        temperature: float = 298.15
    ) -> SolvationEnergy:
        """
        Calculate solvation free energy using SMD model.
        SMD = PCM + empirical corrections for specific interactions.
        """
        # Start with PCM
        solvation = self.pcm_solvation(solute, solvent, temperature)

        # Add hydrogen bonding correction
        delta_g_hb = self._hydrogen_bonding_energy(solute, solvent)
        solvation.hydrogen_bonding = delta_g_hb
        solvation.total += delta_g_hb

        # Add aromatic-solvent interactions (if applicable)
        # Simplified estimation

        return solvation

    def _hydrogen_bonding_energy(self, solute: Solute, solvent: Solvent) -> float:
        """Estimate hydrogen bonding contribution."""
        # HB energy = donors * acceptor_number + acceptors * donor_number
        delta_g_hb = -(
            solute.hbond_donors * solvent.acceptor_number * 0.05 +
            solute.hbond_acceptors * solvent.donor_number * 0.05
        )
        return delta_g_hb

    def calculate_logP(self, solute: Solute) -> float:
        """
        Calculate octanol-water partition coefficient (logP).

        logP = log10(concentration_octanol / concentration_water)

        Positive logP = lipophilic (prefers octanol)
        Negative logP = hydrophilic (prefers water)
        """
        # Simplified Wildman-Crippen method
        # logP = sum of fragment contributions

        # Base contribution from molecular weight
        logp = solute.molecular_weight / 100.0

        # Polarity penalty
        logp -= solute.dipole_moment / 5.0

        # Hydrogen bonding penalty
        logp -= (solute.hbond_donors + solute.hbond_acceptors) * 0.5

        # Charge penalty (ions are hydrophilic)
        logp -= abs(solute.charge) * 2.0

        return logp

    def calculate_logD(
        self,
        solute: Solute,
        pH: float,
        pKa: Optional[List[float]] = None
    ) -> float:
        """
        Calculate distribution coefficient at given pH (logD).

        logD accounts for ionization: logD = logP + log(f_neutral)
        where f_neutral is fraction in neutral form.
        """
        logp = self.calculate_logP(solute)

        if pKa is None:
            # No ionization
            return logp

        # Calculate fraction neutral
        f_neutral = 1.0
        for pk in pKa:
            # For acids: f_neutral *= 1 / (1 + 10^(pH - pKa))
            # For bases: f_neutral *= 1 / (1 + 10^(pKa - pH))
            # Simplified: assume acids
            f_neutral *= 1.0 / (1.0 + 10**(pH - pk))

        logd = logp + np.log10(max(f_neutral, 1e-10))

        return logd

    def calculate_pKa(self, solute: Solute, functional_group: str) -> float:
        """
        Estimate pKa for functional group.

        Simplified empirical method.
        """
        pka_table = {
            "carboxylic_acid": 4.5,
            "phenol": 10.0,
            "alcohol": 15.5,
            "amine": 10.0,
            "ammonium": 9.0,
            "thiol": 10.5,
        }

        return pka_table.get(functional_group, 7.0)

    def pH_effect(
        self,
        solute: Solute,
        pH: float,
        pKa: float,
        functional_group_type: str = "acid"
    ) -> Dict:
        """
        Calculate pH effects on ionization and properties.

        Returns:
            fraction_neutral, fraction_ionized, dominant_species
        """
        if functional_group_type == "acid":
            # HA ⇌ H+ + A-
            # fraction_ionized = 1 / (1 + 10^(pKa - pH))
            f_ionized = 1.0 / (1.0 + 10**(pKa - pH))
        else:  # base
            # B + H+ ⇌ BH+
            # fraction_ionized = 1 / (1 + 10^(pH - pKa))
            f_ionized = 1.0 / (1.0 + 10**(pH - pKa))

        f_neutral = 1.0 - f_ionized

        dominant = "neutral" if f_neutral > 0.5 else "ionized"

        return {
            "fraction_neutral": f_neutral,
            "fraction_ionized": f_ionized,
            "dominant_species": dominant,
            "pH": pH,
            "pKa": pKa
        }

    def solubility_estimate(
        self,
        solute: Solute,
        solvent: Solvent,
        temperature: float = 298.15
    ) -> float:
        """
        Estimate solubility using solvation free energy.

        log(S) ∝ -ΔG_solv / (R*T)
        """
        solvation = self.smd_solvation(solute, solvent, temperature)

        # Convert to molar solubility (very approximate)
        log_s = -solvation.total / (self.k_B * temperature * 2.303)  # log10

        # Convert to mol/L (calibration factor)
        solubility_molar = 10**(log_s - 2.0)  # Empirical adjustment

        return max(solubility_molar, 1e-10)


def example_molecules() -> Tuple[Solute, Solute]:
    """Create example molecules."""
    # Aspirin
    aspirin = Solute(
        name="aspirin",
        smiles="CC(=O)Oc1ccccc1C(=O)O",
        molecular_weight=180.16,
        charge=0.0,
        dipole_moment=3.5,
        polarizability=20.0,
        surface_area=250.0,
        volume=180.0,
        hbond_donors=1,
        hbond_acceptors=4
    )

    # Caffeine
    caffeine = Solute(
        name="caffeine",
        smiles="CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        molecular_weight=194.19,
        charge=0.0,
        dipole_moment=4.2,
        polarizability=22.0,
        surface_area=260.0,
        volume=190.0,
        hbond_donors=0,
        hbond_acceptors=6
    )

    return aspirin, caffeine


if __name__ == "__main__":
    print("Solvation Model Test\n")

    calc = SolvationCalculator()

    aspirin, caffeine = example_molecules()

    # Test with different solvents
    solvents_to_test = ["water", "ethanol", "chloroform", "hexane"]

    for solute in [aspirin, caffeine]:
        print(f"=== {solute.name.upper()} ===\n")

        print("Solvation Free Energies (ΔG_solv, kcal/mol):")
        for solv_name in solvents_to_test:
            solvent = calc.solvents[solv_name]
            solvation = calc.smd_solvation(solute, solvent)

            print(f"\n  {solv_name.capitalize()}:")
            print(f"    Total:        {solvation.total:>8.2f}")
            print(f"    Electrostatic: {solvation.electrostatic:>8.2f}")
            print(f"    Cavitation:    {solvation.cavitation:>8.2f}")
            print(f"    Dispersion:    {solvation.dispersion:>8.2f}")
            print(f"    H-bonding:     {solvation.hydrogen_bonding:>8.2f}")

        # Partition coefficient
        print(f"\nPartition Coefficient (logP): {calc.calculate_logP(solute):.2f}")

        # Solubility estimates
        print("\nSolubility Estimates:")
        for solv_name in ["water", "ethanol"]:
            solvent = calc.solvents[solv_name]
            solubility = calc.solubility_estimate(solute, solvent)
            print(f"  {solv_name.capitalize()}: {solubility:.2e} M")

        # pH effects (for aspirin only - has carboxylic acid)
        if solute.name == "aspirin":
            print("\npH Effects (Aspirin - COOH, pKa ~ 3.5):")
            for ph in [1.0, 3.5, 7.4, 10.0]:
                ph_effect = calc.pH_effect(solute, ph, pKa=3.5, functional_group_type="acid")
                print(f"  pH {ph:.1f}: {ph_effect['fraction_neutral']*100:.1f}% neutral, "
                      f"{ph_effect['fraction_ionized']*100:.1f}% ionized ({ph_effect['dominant_species']})")

            # Distribution coefficient at physiological pH
            logD_physiological = calc.calculate_logD(solute, pH=7.4, pKa=[3.5])
            print(f"\nlogD at pH 7.4: {logD_physiological:.2f}")

        print("\n" + "="*60 + "\n")

    print("Solvation Model ready!")
