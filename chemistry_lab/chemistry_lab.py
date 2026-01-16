"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Chemistry Laboratory - Main API
Unified interface for all chemistry laboratory capabilities.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from numbers import Real
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING, Any

# Import all sub-modules
from .molecular_dynamics import (
    MolecularDynamics, ForceField, Integrator, Ensemble,
    Atom as MDAtom, MDState, create_water_box
)
from .reaction_simulator import (
    ReactionSimulator, ReactionPath, ReactionConditions,
    Catalyst, Molecule as ReactMolecule, ReactionType
)
from .synthesis_planner import (
    SynthesisPlanner, SynthesisRoute, Compound, Transformation
)
from .spectroscopy_predictor import (
    SpectroscopyPredictor, Spectrum, SpectroscopyType
)
from .solvation_model import (
    SolvationCalculator, Solute, Solvent, SolvationEnergy
)
from .quantum_chemistry_interface import (
    QuantumChemistryInterface, QMMethod, BasisSet, DFTFunctional,
    Molecule as QMMolecule, QMResult
)

try:
    from core.base_lab import BaseLab
except ImportError:
    # This is a fallback for script execution
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from core.base_lab import BaseLab


if TYPE_CHECKING:
    from materials_lab.materials_lab import MaterialsLab
    from environmental_sim.environmental_sim import EnvironmentalSimulator


@dataclass
class ChemistryLabConfig:
    """Configuration for chemistry laboratory."""
    enable_md: bool = True
    enable_reactions: bool = True
    enable_synthesis: bool = True
    enable_spectroscopy: bool = True
    enable_solvation: bool = True
    enable_quantum: bool = True

    default_force_field: ForceField = ForceField.AMBER
    default_qm_method: QMMethod = QMMethod.DFT
    default_basis_set: BasisSet = BasisSet.SIX_31G_STAR

    temperature: float = 298.15  # K
    pressure: float = 1.0  # bar


class ChemistryLaboratory(BaseLab):
    """
    Unified chemistry laboratory interface.

    Provides access to:
    1. Molecular Dynamics (100k atoms @ 1fs)
    2. Reaction Simulation (TST, NEB, catalysis)
    3. Synthesis Planning (retrosynthesis, optimization)
    4. Spectroscopy Prediction (NMR, IR, UV-Vis, MS, XRD)
    5. Solvation Models (PCM, COSMO, SMD, pH effects)
    6. Quantum Chemistry (DFT, HF, MP2, CCSD(T))

    Target accuracy: <5% error on reaction energetics, <10% on spectroscopy
    """

    def __init__(self, config: Optional[ChemistryLabConfig] = None):
        if isinstance(config, dict):
            # If a dict is passed, create a config object from it
            config_obj = ChemistryLabConfig(**config)
            super().__init__(config)
            self.config = config_obj
        else:
            config_obj = config or ChemistryLabConfig()
            super().__init__(vars(config_obj))
            self.config = config_obj

        # Initialize sub-systems
        self.md_engine = None
        self.reaction_sim = ReactionSimulator() if self.config.enable_reactions else None
        self.synthesis_planner = SynthesisPlanner() if self.config.enable_synthesis else None
        self.spectroscopy = SpectroscopyPredictor() if self.config.enable_spectroscopy else None
        self.solvation = SolvationCalculator() if self.config.enable_solvation else None
        self.quantum = QuantumChemistryInterface() if self.config.enable_quantum else None

        print("[ChemLab] Chemistry Laboratory initialized")
        self._print_capabilities()

    def run_experiment(self, experiment_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a chemistry experiment.
        """
        exp_type = experiment_spec.get("experiment_type")
        if not exp_type:
            raise ValueError("'experiment_type' is required.")

        if exp_type == "md_simulation":
            # Simplified example - a real implementation would deserialize atoms etc.
            atoms, _, _ = create_water_box(experiment_spec.get("num_atoms", 100), box_size=15.0)
            self.create_md_simulation(atoms, np.array([15.0, 15.0, 15.0]))
            trajectory = self.run_md_simulation(n_steps=experiment_spec.get("n_steps", 100))
            return {"status": "completed", "frames": len(trajectory)}
        
        elif exp_type == "reaction_simulation":
            reactants_spec = experiment_spec.get("reactants")
            products_spec = experiment_spec.get("products")
            conditions_spec = experiment_spec.get("conditions", {})

            self._validate_reaction_inputs(reactants_spec, products_spec, conditions_spec)

            reactants = self._build_reactant_molecules(reactants_spec)
            products = self._build_reactant_molecules(products_spec)
            conditions = self._build_reaction_conditions(conditions_spec)
            reaction_name = experiment_spec.get("reaction_name")

            result = self.simulate_reaction(
                reactants,
                products,
                conditions,
                reaction_name=reaction_name
            )

            kinetics = result.get("kinetics")
            profiles = {
                "time": result["time"].tolist(),
                "reactant_concentration": result["reactant_concentration"].tolist(),
                "product_concentration": result["product_concentration"].tolist(),
                "product_distribution": {
                    key: value.tolist() for key, value in result["product_distribution"].items()
                }
            }

            return {
                "status": "completed",
                "reaction_name": reaction_name or "custom_reaction",
                "kinetics": asdict(kinetics) if kinetics else {},
                "selectivity": kinetics.product_selectivity if kinetics else {},
                "profiles": profiles,
            }

        elif exp_type == "spectroscopy":
            molecule = experiment_spec.get("molecule")
            if not molecule:
                raise ValueError("'molecule' is required for spectroscopy.")
            spectrum_type = experiment_spec.get("spectrum_type", "nmr_1h")
            if spectrum_type == "nmr_1h":
                result = self.predict_nmr(molecule, "1H")
            elif spectrum_type == "ir":
                result = self.predict_ir(molecule)
            else:
                raise ValueError(f"Unsupported spectrum type: {spectrum_type}")
            return result.to_dict()
            
        else:
            raise ValueError(f"Unknown experiment type: {exp_type}")

    def get_status(self) -> Dict[str, Any]:
        """Returns a summary of the lab's configuration and capabilities."""
        caps = {}
        if self.config.enable_md: caps["molecular_dynamics"] = True
        if self.config.enable_reactions: caps["reaction_simulation"] = True
        if self.config.enable_synthesis: caps["synthesis_planning"] = True
        if self.config.enable_spectroscopy: caps["spectroscopy_prediction"] = True
        if self.config.enable_solvation: caps["solvation_models"] = True
        if self.config.enable_quantum: caps["quantum_chemistry"] = True
        
        return {
            "lab_name": self.__class__.__name__,
            "capabilities": caps,
            "default_force_field": self.config.default_force_field.value,
            "default_qm_method": self.config.default_qm_method.value,
        }

    def _print_capabilities(self):
        """Print available capabilities."""
        print("\nAvailable capabilities:")
        if self.config.enable_md:
            print("  ✓ Molecular Dynamics (AMBER, CHARMM, OPLS)")
        if self.config.enable_reactions:
            print("  ✓ Reaction Simulation (TST, NEB, kinetics)")
        if self.config.enable_synthesis:
            print("  ✓ Synthesis Planning (retrosynthesis)")
        if self.config.enable_spectroscopy:
            print("  ✓ Spectroscopy Prediction (NMR, IR, UV, MS, XRD)")
        if self.config.enable_solvation:
            print("  ✓ Solvation Models (PCM, SMD, logP)")
        if self.config.enable_quantum:
            print("  ✓ Quantum Chemistry (DFT, HF, MP2)")
        print()

    # ========== Molecular Dynamics ==========

    def create_md_simulation(
        self,
        atoms: List[MDAtom],
        box_size: np.ndarray,
        force_field: Optional[ForceField] = None,
        ensemble: Ensemble = Ensemble.NVT,
        timestep: float = 1.0
    ) -> MolecularDynamics:
        """Create molecular dynamics simulation."""
        if not self.config.enable_md:
            raise RuntimeError("Molecular dynamics disabled")

        ff = force_field or self.config.default_force_field

        self.md_engine = MolecularDynamics(
            atoms=atoms,
            box_size=box_size,
            force_field=ff,
            integrator=Integrator.VERLET,
            ensemble=ensemble,
            timestep=timestep
        )

        return self.md_engine

    def run_md_simulation(
        self,
        n_steps: int,
        temperature: Optional[float] = None,
        output_interval: int = 100
    ) -> List[MDState]:
        """Run MD simulation."""
        if self.md_engine is None:
            raise RuntimeError("MD engine not initialized. Call create_md_simulation first.")

        temp = temperature or self.config.temperature
        self.md_engine.set_temperature(temp)

        trajectory = self.md_engine.run(n_steps, output_interval)

        return trajectory

    # ========== Reaction Simulation ==========

    def simulate_reaction(
        self,
        reactants: List[ReactMolecule],
        products: List[ReactMolecule],
        conditions: Optional[ReactionConditions] = None,
        reaction_name: Optional[str] = None,
        materials_lab: Optional["MaterialsLab"] = None,
        environment_simulator: Optional["EnvironmentalSimulator"] = None
    ) -> Dict:
        """Simulate chemical reaction and predict kinetics."""
        if not self.config.enable_reactions:
            raise RuntimeError("Reaction simulation disabled")

        # Find reaction pathway
        path = self.reaction_sim.nudged_elastic_band(reactants, products)

        # Set default conditions
        if conditions is None:
            conditions = ReactionConditions(
                temperature=self.config.temperature,
                pressure=self.config.pressure
            )

        metadata = self.reaction_sim.get_reaction_metadata(reaction_name)

        # Predict kinetics
        kinetics = self.reaction_sim.predict_reaction_kinetics(
            path,
            conditions,
            reaction_name=reaction_name,
            metadata=metadata
        )

        # Simulate concentration profiles
        time, reactant_conc, product_conc, product_profiles = self.reaction_sim.simulate_reaction_kinetics(
            path,
            conditions,
            reaction_name=reaction_name,
            metadata=metadata,
            return_profiles=True
        )

        result = {
            "pathway": path,
            "kinetics": kinetics,
            "time": time,
            "reactant_concentration": reactant_conc,
            "product_concentration": product_conc,
            "product_distribution": product_profiles,
            "selectivity": kinetics.product_selectivity,
        }

        if metadata:
            integration_payload = self.reaction_sim.build_integration_payload(metadata, kinetics, conditions)
            result["integration"] = integration_payload

            # Optional automatic propagation
            if materials_lab is not None:
                from .integration import apply_material_updates
                apply_material_updates(materials_lab, integration_payload.get("materials", {}))

            if environment_simulator is not None:
                from .integration import apply_environmental_adjustments
                apply_environmental_adjustments(environment_simulator, integration_payload.get("environment", []))

        return result

    def get_reaction_metadata(self, reaction_name: str) -> Optional[Dict[str, Any]]:
        """Expose reaction metadata summary."""
        metadata = self.reaction_sim.get_reaction_metadata(reaction_name)
        return metadata.to_summary() if metadata else None

    # ========== Synthesis Planning ==========

    def plan_synthesis(
        self,
        target: Compound,
        max_depth: int = 5
    ) -> SynthesisRoute:
        """Plan synthesis route to target compound."""
        if not self.config.enable_synthesis:
            raise RuntimeError("Synthesis planning disabled")

        # Perform retrosynthesis
        tree = self.synthesis_planner.retrosynthesis(target, max_depth)

        # Extract routes
        routes = self.synthesis_planner.extract_routes(tree)

        if not routes:
            raise ValueError("No viable synthesis routes found")

        # Optimize
        best_route = self.synthesis_planner.optimize_route(routes)

        return best_route

    def analyze_synthesis_safety(self, route: SynthesisRoute) -> Dict:
        """Analyze safety of synthesis route."""
        if not self.config.enable_synthesis:
            raise RuntimeError("Synthesis planning disabled")

        return self.synthesis_planner.safety_analysis(route)

    # ========== Spectroscopy ==========

    def predict_nmr(
        self,
        molecule: Dict,
        nucleus: str = "1H"
    ) -> Spectrum:
        """Predict NMR spectrum."""
        if not self.config.enable_spectroscopy:
            raise RuntimeError("Spectroscopy disabled")

        if nucleus == "1H":
            return self.spectroscopy.predict_nmr_1h(molecule)
        elif nucleus == "13C":
            return self.spectroscopy.predict_nmr_13c(molecule)
        else:
            raise ValueError(f"Unsupported nucleus: {nucleus}")

    def predict_ir(self, molecule: Dict) -> Spectrum:
        """Predict IR spectrum."""
        if not self.config.enable_spectroscopy:
            raise RuntimeError("Spectroscopy disabled")

        return self.spectroscopy.predict_ir(molecule)

    def predict_uv_vis(self, molecule: Dict) -> Spectrum:
        """Predict UV-Vis spectrum."""
        if not self.config.enable_spectroscopy:
            raise RuntimeError("Spectroscopy disabled")

        return self.spectroscopy.predict_uv_vis(molecule)

    def predict_mass_spectrum(self, molecule: Dict) -> Spectrum:
        """Predict mass spectrum."""
        if not self.config.enable_spectroscopy:
            raise RuntimeError("Spectroscopy disabled")

        return self.spectroscopy.predict_mass_spec(molecule)

    # ========== Solvation ==========

    def calculate_solvation_energy(
        self,
        solute: Solute,
        solvent_name: str,
        model: str = "smd"
    ) -> SolvationEnergy:
        """Calculate solvation free energy."""
        if not self.config.enable_solvation:
            raise RuntimeError("Solvation disabled")

        solvent = self.solvation.solvents.get(solvent_name)
        if not solvent:
            raise ValueError(f"Unknown solvent: {solvent_name}")

        if model == "pcm":
            return self.solvation.pcm_solvation(solute, solvent, self.config.temperature)
        elif model == "smd":
            return self.solvation.smd_solvation(solute, solvent, self.config.temperature)
        else:
            raise ValueError(f"Unknown solvation model: {model}")

    def predict_logP(self, solute: Solute) -> float:
        """Predict octanol-water partition coefficient."""
        if not self.config.enable_solvation:
            raise RuntimeError("Solvation disabled")

        return self.solvation.calculate_logP(solute)

    def calculate_pH_effect(
        self,
        solute: Solute,
        pH: float,
        pKa: float
    ) -> Dict:
        """Calculate pH effects on ionization."""
        if not self.config.enable_solvation:
            raise RuntimeError("Solvation disabled")

        return self.solvation.pH_effect(solute, pH, pKa)

    # ========== Quantum Chemistry ==========

    def quantum_calculation(
        self,
        molecule: QMMolecule,
        method: Optional[QMMethod] = None,
        basis_set: Optional[BasisSet] = None
    ) -> QMResult:
        """Perform quantum chemistry calculation."""
        if not self.config.enable_quantum:
            raise RuntimeError("Quantum chemistry disabled")

        qm_method = method or self.config.default_qm_method
        bs = basis_set or self.config.default_basis_set

        if qm_method == QMMethod.HF:
            return self.quantum.hartree_fock(molecule, bs)
        elif qm_method == QMMethod.DFT:
            return self.quantum.dft(molecule, DFTFunctional.B3LYP, bs)
        elif qm_method == QMMethod.MP2:
            return self.quantum.mp2(molecule, bs)
        else:
            return self.quantum.dft(molecule, DFTFunctional.B3LYP, bs)

    def optimize_geometry(
        self,
        molecule: QMMolecule,
        method: Optional[QMMethod] = None
    ) -> Tuple[QMMolecule, QMResult]:
        """Optimize molecular geometry."""
        if not self.config.enable_quantum:
            raise RuntimeError("Quantum chemistry disabled")

        qm_method = method or self.config.default_qm_method

        return self.quantum.optimize_geometry(molecule, qm_method, self.config.default_basis_set)

    def calculate_vibrational_frequencies(
        self,
        molecule: QMMolecule
    ) -> QMResult:
        """Calculate vibrational frequencies."""
        if not self.config.enable_quantum:
            raise RuntimeError("Quantum chemistry disabled")

        return self.quantum.calculate_vibrational_frequencies(
            molecule, self.config.default_qm_method, self.config.default_basis_set
        )

    # ========== Integrated Workflows ==========

    def complete_molecule_characterization(self, molecule: Dict) -> Dict:
        """
        Complete characterization of molecule:
        - Spectroscopy (NMR, IR, UV-Vis, MS)
        - Solvation (logP, solubility)
        - Quantum properties (HOMO-LUMO, dipole)
        """
        results = {}

        # Spectroscopy
        if self.config.enable_spectroscopy:
            results["nmr_1h"] = self.predict_nmr(molecule, "1H")
            results["nmr_13c"] = self.predict_nmr(molecule, "13C")
            results["ir"] = self.predict_ir(molecule)
            results["uv_vis"] = self.predict_uv_vis(molecule)
            results["mass_spec"] = self.predict_mass_spectrum(molecule)

        # Solvation (if Solute object provided)
        if self.config.enable_solvation and "solute" in molecule:
            solute = molecule["solute"]
            results["logP"] = self.predict_logP(solute)

            # Solvation in water
            results["solvation_water"] = self.calculate_solvation_energy(
                solute, "water", "smd"
            )

        return results

    def _validate_reaction_inputs(
        self,
        reactants: Any,
        products: Any,
        conditions: Any
    ) -> None:
        """Minimal validation for reaction experiment inputs."""
        if not isinstance(reactants, list) or not reactants:
            raise ValueError("'reactants' must be a non-empty list of molecule specifications.")
        if not isinstance(products, list) or not products:
            raise ValueError("'products' must be a non-empty list of molecule specifications.")
        if not isinstance(conditions, dict):
            raise ValueError("'conditions' must be provided as a dictionary.")

        for label, collection in (("reactant", reactants), ("product", products)):
            for idx, entry in enumerate(collection):
                if not isinstance(entry, dict):
                    raise ValueError(f"Each {label} specification must be a dictionary (item {idx}).")
                if not (entry.get("formula") or entry.get("smiles")):
                    raise ValueError(f"{label.capitalize()} {idx + 1} requires 'formula' or 'smiles'.")
                for field in ("energy", "enthalpy", "entropy"):
                    if field not in entry:
                        raise ValueError(f"{label.capitalize()} {idx + 1} missing required field '{field}'.")
                    if not isinstance(entry[field], Real):
                        raise ValueError(f"{label.capitalize()} {idx + 1} field '{field}' must be numeric.")

        self._validate_conditions(
            conditions.get("temperature", self.config.temperature),
            conditions.get("pressure", self.config.pressure),
            conditions.get("pH")
        )

    def _validate_conditions(self, temperature: Any, pressure: Any, pH: Any = None) -> None:
        """Validate basic reaction conditions."""
        if not isinstance(temperature, Real) or temperature <= 0:
            raise ValueError("'temperature' must be a positive number in Kelvin.")
        if not isinstance(pressure, Real) or pressure <= 0:
            raise ValueError("'pressure' must be a positive number in bar.")
        if pH is not None and (not isinstance(pH, Real) or pH < 0 or pH > 14):
            raise ValueError("'pH' must be between 0 and 14 if provided.")

    def _build_reaction_conditions(self, conditions_spec: Dict[str, Any]) -> ReactionConditions:
        """Construct ReactionConditions from a dictionary specification."""
        temperature = conditions_spec.get("temperature", self.config.temperature)
        pressure = conditions_spec.get("pressure", self.config.pressure)
        pH = conditions_spec.get("pH")
        catalyst = conditions_spec.get("catalyst")

        if catalyst is not None and not isinstance(catalyst, Catalyst):
            raise ValueError("'catalyst' must be a Catalyst instance if provided.")

        self._validate_conditions(temperature, pressure, pH)

        return ReactionConditions(
            temperature=temperature,
            pressure=pressure,
            solvent=conditions_spec.get("solvent"),
            pH=pH,
            catalyst=catalyst
        )

    def _build_reactant_molecules(self, molecules_spec: List[Dict[str, Any]]) -> List[ReactMolecule]:
        """Convert dictionary specifications into ReactMolecule instances."""
        molecules: List[ReactMolecule] = []
        for index, spec in enumerate(molecules_spec):
            formula = spec.get("formula", "")
            smiles = spec.get("smiles", "")
            geometry = spec.get("geometry")

            try:
                molecule = ReactMolecule(
                    formula=formula,
                    smiles=smiles,
                    energy=float(spec["energy"]),
                    enthalpy=float(spec["enthalpy"]),
                    entropy=float(spec["entropy"]),
                    geometry=geometry
                )
            except KeyError as exc:
                raise ValueError(f"Molecule {index + 1} missing required key: {exc.args[0]}") from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Molecule {index + 1} has invalid numeric values: {exc}") from exc

            molecules.append(molecule)

        return molecules

    def reaction_optimization_workflow(
        self,
        reactants: List[ReactMolecule],
        products: List[ReactMolecule],
        catalysts: Optional[List[Catalyst]] = None
    ) -> Dict:
        """
        Optimize reaction conditions:
        - Find optimal catalyst
        - Determine best temperature/pressure
        - Predict yields
        """
        results = {}

        # Baseline reaction
        baseline_conditions = ReactionConditions(
            temperature=self.config.temperature,
            pressure=self.config.pressure
        )

        baseline = self.simulate_reaction(reactants, products, baseline_conditions)
        results["baseline"] = baseline

        # Test catalysts if provided
        if catalysts:
            catalyst_results = []
            for catalyst in catalysts:
                cat_conditions = ReactionConditions(
                    temperature=self.config.temperature,
                    pressure=self.config.pressure,
                    catalyst=catalyst
                )
                cat_result = self.simulate_reaction(reactants, products, cat_conditions)
                catalyst_results.append({
                    "catalyst": catalyst.name,
                    "result": cat_result
                })

            results["catalysts"] = catalyst_results

        # Temperature scan
        temp_results = []
        for temp in [273, 298, 323, 373]:
            temp_conditions = ReactionConditions(temperature=temp, pressure=self.config.pressure)
            temp_result = self.simulate_reaction(reactants, products, temp_conditions)
            temp_results.append({
                "temperature": temp,
                "result": temp_result
            })

        results["temperature_scan"] = temp_results

        return results


if __name__ == "__main__":
    print("=" * 80)
    print("CHEMISTRY LABORATORY - MAIN API TEST")
    print("=" * 80)
    print()

    # Initialize laboratory
    lab = ChemistryLaboratory()

    # Test 1: Molecular Dynamics
    print("\n[TEST 1] Molecular Dynamics")
    print("-" * 40)
    atoms, bonds, angles = create_water_box(100, box_size=15.0)
    md = lab.create_md_simulation(
        atoms=atoms,
        box_size=np.array([15.0, 15.0, 15.0]),
        ensemble=Ensemble.NVT
    )
    trajectory = lab.run_md_simulation(n_steps=100, temperature=300.0, output_interval=50)
    print(f"MD simulation completed: {len(trajectory)} frames")
    print(f"Final temperature: {trajectory[-1].temperature:.2f} K")
    print(f"Final energy: {trajectory[-1].potential_energy + trajectory[-1].kinetic_energy:.2f} kcal/mol")

    # Test 2: Spectroscopy
    print("\n[TEST 2] Spectroscopy Prediction")
    print("-" * 40)
    aspirin = {
        'name': 'aspirin',
        'smiles': 'CC(=O)Oc1ccccc1C(=O)O',
        'molecular_weight': 180.16,
        'functional_groups': ['ester', 'carboxylic_acid', 'aromatic']
    }

    nmr = lab.predict_nmr(aspirin, "1H")
    print(f"1H NMR: {len(nmr.peaks)} peaks")

    ir = lab.predict_ir(aspirin)
    print(f"IR: {len(ir.peaks)} peaks")

    # Test 3: Solvation
    print("\n[TEST 3] Solvation Calculations")
    print("-" * 40)
    aspirin_solute = Solute(
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

    solvation = lab.calculate_solvation_energy(aspirin_solute, "water", "smd")
    print(f"Solvation energy (water): {solvation.total:.2f} kcal/mol")

    logp = lab.predict_logP(aspirin_solute)
    print(f"logP: {logp:.2f}")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
