"""
from nist_constants import *

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Reaction Simulator
Simulate chemical reactions with transition state theory, reaction pathways, catalysis, and kinetics.
"""

# NOTE: The phrase "free energy" in this module refers exclusively to the Gibbs
# free energy used in standard thermodynamics calculations, not pseudoscience.

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np


class ReactionType(Enum):
    """Types of chemical reactions."""
    SYNTHESIS = "synthesis"
    DECOMPOSITION = "decomposition"
    SUBSTITUTION = "substitution"
    ADDITION = "addition"
    ELIMINATION = "elimination"
    REDOX = "redox"
    CATALYTIC = "catalytic"


@dataclass
class ReactionSolventEffect:
    """Effect of solvent on kinetics/selectivity."""
    solvent: str
    dielectric: float
    rate_factor: float = 1.0
    selectivity_shift: float = 0.0
    selectivity_profile: Dict[str, float] = field(default_factory=dict)

    def matches(self, solvent_name: Optional[str]) -> bool:
        return bool(solvent_name) and solvent_name.strip().lower() == self.solvent.lower()


@dataclass
class ReactionByproductInfo:
    """Byproduct metadata for integration and safety."""
    name: str
    material_id: str
    yield_fraction: float = 0.0
    hazards: List[str] = field(default_factory=list)
    disposal: str = "standard"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "material_id": self.material_id,
            "yield_fraction": self.yield_fraction,
            "hazards": list(self.hazards),
            "disposal": self.disposal,
        }


@dataclass
class ReactionSafetyProfile:
    """Safety limits and PPE guidance."""
    hazards: List[str] = field(default_factory=list)
    ppe: List[str] = field(default_factory=list)
    flash_point_c: Optional[float] = None
    toxicity_ld50_mg_per_kg: Optional[float] = None
    temperature_limits_c: Tuple[Optional[float], Optional[float]] = (None, None)
    pressure_limits_bar: Tuple[Optional[float], Optional[float]] = (None, None)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hazards": list(self.hazards),
            "ppe": list(self.ppe),
            "flash_point_c": self.flash_point_c,
            "toxicity_ld50_mg_per_kg": self.toxicity_ld50_mg_per_kg,
            "temperature_limits_c": self.temperature_limits_c,
            "pressure_limits_bar": self.pressure_limits_bar,
        }


@dataclass
class ReactionMaterialsLink:
    """Link to materials database identifiers."""
    reactant_ids: List[str] = field(default_factory=list)
    product_ids: List[str] = field(default_factory=list)
    byproduct_ids: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "reactant_ids": list(self.reactant_ids),
            "product_ids": list(self.product_ids),
            "byproduct_ids": list(self.byproduct_ids),
        }


@dataclass
class ReactionKineticsParameters:
    """Stored kinetics parameters from database."""
    arrhenius_A: float
    temperature_exponent: float
    activation_energy_kcal_per_mol: float
    rate_units: str
    reaction_order: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "arrhenius_A": self.arrhenius_A,
            "temperature_exponent": self.temperature_exponent,
            "activation_energy_kcal_per_mol": self.activation_energy_kcal_per_mol,
            "rate_units": self.rate_units,
            "reaction_order": self.reaction_order,
        }


@dataclass
class ReactionThermodynamicsParameters:
    """Stored thermodynamic parameters from database."""
    delta_h_kcal_per_mol: Optional[float] = None
    delta_s_cal_per_mol_k: Optional[float] = None
    delta_g_kcal_per_mol: Optional[float] = None
    heat_capacity_change_cal_per_mol_k: Optional[float] = None
    equilibrium_constant_log10: Optional[float] = None

    def equilibrium_constant(self, temperature: float, gas_constant_kcal: float) -> float:
        if self.equilibrium_constant_log10 is not None:
            return 10 ** self.equilibrium_constant_log10
        if self.delta_g_kcal_per_mol is not None:
            return float(np.exp(-self.delta_g_kcal_per_mol / (gas_constant_kcal * temperature)))
        return 1.0

    def ensure_delta_g(self, temperature: float) -> None:
        if self.delta_g_kcal_per_mol is None and self.delta_h_kcal_per_mol is not None and self.delta_s_cal_per_mol_k is not None:
            self.delta_g_kcal_per_mol = self.delta_h_kcal_per_mol - temperature * self.delta_s_cal_per_mol_k / 1000.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "delta_h_kcal_per_mol": self.delta_h_kcal_per_mol,
            "delta_s_cal_per_mol_k": self.delta_s_cal_per_mol_k,
            "delta_g_kcal_per_mol": self.delta_g_kcal_per_mol,
            "heat_capacity_change_cal_per_mol_k": self.heat_capacity_change_cal_per_mol_k,
            "equilibrium_constant_log10": self.equilibrium_constant_log10,
        }


@dataclass
class ReactionMetadata:
    """Complete metadata record for a catalogued reaction."""
    name: str
    type: str
    reactants: List[str]
    products: List[str]
    materials: ReactionMaterialsLink
    kinetics: ReactionKineticsParameters
    thermodynamics: ReactionThermodynamicsParameters
    solvent_effects: List[ReactionSolventEffect] = field(default_factory=list)
    byproducts: List[ReactionByproductInfo] = field(default_factory=list)
    safety: ReactionSafetyProfile = field(default_factory=ReactionSafetyProfile)
    references: List[str] = field(default_factory=list)
    material_effects: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    environment_effects: List[Dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "reactants": self.reactants,
            "products": self.products,
            "materials": self.materials.as_dict(),
            "kinetics": self.kinetics.as_dict(),
            "thermodynamics": self.thermodynamics.as_dict(),
            "solvent_effects": [effect.__dict__ for effect in self.solvent_effects],
            "byproducts": [bp.as_dict() for bp in self.byproducts],
            "safety": self.safety.as_dict(),
            "references": list(self.references),
            "material_effects": self.material_effects,
            "environment_effects": [dict(effect) for effect in self.environment_effects],
        }


@dataclass
class Molecule:
    """Molecule representation."""
    formula: str
    smiles: str
    energy: float  # kcal/mol (relative energy)
    enthalpy: float  # kcal/mol
    entropy: float  # cal/(mol*K)
    geometry: Optional[np.ndarray] = None  # Atom positions


@dataclass
class TransitionState:
    """Transition state between reactants and products."""
    geometry: np.ndarray
    energy: float  # kcal/mol
    imaginary_frequency: float  # cm^-1 (negative for TS)
    reaction_coordinate: np.ndarray


@dataclass
class ReactionPath:
    """Complete reaction pathway from reactants to products."""
    reactants: List[Molecule]
    products: List[Molecule]
    transition_states: List[TransitionState]
    intermediates: List[Molecule]
    barriers_forward: List[float]  # Activation energies (kcal/mol)
    barriers_reverse: List[float]
    reaction_energy: float  # ΔE (kcal/mol)
    reaction_enthalpy: float  # ΔH (kcal/mol)
    reaction_entropy: float  # ΔS (cal/(mol*K))
    reaction_gibbs: float  # ΔG (kcal/mol)


@dataclass
class Catalyst:
    """Catalyst information."""
    name: str
    formula: str
    active_sites: List[str]
    barrier_reduction: float  # kcal/mol (how much it lowers barrier)
    selectivity: Dict[str, float]  # Product selectivity


@dataclass
class ReactionConditions:
    """Reaction conditions."""
    temperature: float  # K
    pressure: float  # bar
    solvent: Optional[str] = None
    pH: Optional[float] = None
    catalyst: Optional[Catalyst] = None


@dataclass
class ReactionKinetics:
    """Reaction kinetics data."""
    rate_constant: float  # s^-1 or M^-1*s^-1
    activation_energy: float  # kcal/mol
    pre_exponential_factor: float  # A in Arrhenius equation
    reaction_order: int
    half_life: float  # s
    equilibrium_constant: float
    product_selectivity: Dict[str, float] = field(default_factory=dict)


class ReactionSimulator:
    """
    Simulate chemical reactions with transition state theory.

    Features:
    - Transition state theory (TST) for rate calculations
    - Nudged Elastic Band (NEB) for reaction path finding
    - Reaction barrier calculations
    - Catalysis effects
    - Kinetics and equilibrium
    - Temperature/pressure/solvent effects
    """

    def __init__(self):
        self.R = 1.987204e-3  # Gas constant in kcal/(mol*K)
        self.k_B = 1.380649e-23  # Boltzmann constant in J/K
        self.h = 6.62607015e-34  # Planck constant in J*s
        self.default_initial_concentration = 1.0  # M
        self._reaction_catalog = self._load_reactions_database()
        # Legacy dictionary for backward compatibility
        self.reactions_database = {
            name: metadata.to_summary()
            for name, metadata in self._reaction_catalog.items()
        }
        self._last_selectivity_profile: Dict[str, float] = {}
        self._last_product_profiles: Dict[str, np.ndarray] = {}

    def _load_reactions_database(self) -> Dict[str, ReactionMetadata]:
        """Load reaction metadata from JSON database."""
        database_path = Path(__file__).resolve().parent / "data" / "reaction_database.json"
        if not database_path.exists():
            return {}

        with database_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        catalog: Dict[str, ReactionMetadata] = {}
        reactions = payload.get("reactions", [])

        for entry in reactions:
            try:
                materials = entry.get("materials", {})
                kinetics = entry.get("kinetics", {})
                thermodynamics = entry.get("thermodynamics", {})
                safety = entry.get("safety", {})

                metadata = ReactionMetadata(
                    name=entry["name"],
                    type=entry.get("type", "unknown"),
                    reactants=list(entry.get("reactants", [])),
                    products=list(entry.get("products", [])),
                    materials=ReactionMaterialsLink(
                        reactant_ids=list(materials.get("reactant_ids", [])),
                        product_ids=list(materials.get("product_ids", [])),
                        byproduct_ids=list(materials.get("byproduct_ids", [])),
                    ),
                    kinetics=ReactionKineticsParameters(
                        arrhenius_A=kinetics.get("arrhenius_A", 1e13),
                        temperature_exponent=kinetics.get("arrhenius_n", 0.0),
                        activation_energy_kcal_per_mol=kinetics.get("activation_energy_kcal_per_mol", 20.0),
                        rate_units=kinetics.get("rate_units", "s^-1"),
                        reaction_order=kinetics.get("reaction_order", 1),
                    ),
                    thermodynamics=ReactionThermodynamicsParameters(
                        delta_h_kcal_per_mol=thermodynamics.get("delta_h_kcal_per_mol"),
                        delta_s_cal_per_mol_k=thermodynamics.get("delta_s_cal_per_mol_k"),
                        delta_g_kcal_per_mol=thermodynamics.get("delta_g_kcal_per_mol"),
                        heat_capacity_change_cal_per_mol_k=thermodynamics.get("heat_capacity_change_cal_per_mol_k"),
                        equilibrium_constant_log10=thermodynamics.get("equilibrium_constant_log10"),
                    ),
                    solvent_effects=[
                        ReactionSolventEffect(
                            solvent=effect.get("solvent", ""),
                            dielectric=effect.get("dielectric", 0.0),
                            rate_factor=effect.get("rate_factor", 1.0),
                            selectivity_shift=effect.get("selectivity_shift", 0.0),
                            selectivity_profile=dict(effect.get("selectivity_profile", {})),
                        )
                        for effect in entry.get("solvent_effects", [])
                    ],
                    byproducts=[
                        ReactionByproductInfo(
                            name=bp.get("name", ""),
                            material_id=bp.get("material_id", ""),
                            yield_fraction=bp.get("yield_fraction", 0.0),
                            hazards=list(bp.get("hazards", [])),
                            disposal=bp.get("disposal", "standard"),
                        )
                        for bp in entry.get("byproducts", [])
                    ],
                    safety=ReactionSafetyProfile(
                        hazards=list(safety.get("hazards", [])),
                        ppe=list(safety.get("ppe", [])),
                        flash_point_c=safety.get("flash_point_c"),
                        toxicity_ld50_mg_per_kg=safety.get("toxicity_ld50_mg_per_kg"),
                        temperature_limits_c=tuple(safety.get("temperature_limits_c", (None, None))),
                        pressure_limits_bar=tuple(safety.get("pressure_limits_bar", (None, None))),
                    ),
                    references=list(entry.get("references", [])),
                    material_effects={
                        name: {
                            prop: dict(spec)
                            for prop, spec in effects.items()
                        }
                        for name, effects in entry.get("material_effects", {}).items()
                    },
                    environment_effects=[dict(effect) for effect in entry.get("environment_effects", [])],
                )

                metadata.thermodynamics.ensure_delta_g(298.15)
                catalog[metadata.name.lower()] = metadata
            except KeyError as exc:
                raise ValueError(f"Invalid reaction database entry: missing key {exc}") from exc

        return catalog

    def list_database_reactions(self) -> List[str]:
        """List available reactions in the metadata catalog."""
        return sorted(self._reaction_catalog.keys())

    def get_reaction_metadata(self, name: Optional[str]) -> Optional[ReactionMetadata]:
        """Retrieve reaction metadata by name (case-insensitive)."""
        if not name:
            return None
        return self._reaction_catalog.get(name.lower())

    @staticmethod
    def _resolve_product_labels(metadata: Optional[ReactionMetadata], path: ReactionPath) -> List[str]:
        """Resolve product labels from metadata or reaction pathway."""
        if metadata and metadata.products:
            return list(metadata.products)

        labels: List[str] = []
        for index, molecule in enumerate(path.products):
            label = getattr(molecule, "formula", None) or getattr(molecule, "smiles", None)
            if not label:
                label = f"product_{index + 1}"
            labels.append(label)

        if not labels:
            labels.append("product")
        return labels

    @staticmethod
    def _normalize_distribution(distribution: Dict[str, Any], fallback_keys: List[str]) -> Dict[str, float]:
        """Normalize selectivity distribution, falling back to uniform if needed."""
        positive: Dict[str, float] = {}
        for key, value in distribution.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric > 0:
                positive[key] = numeric

        total = sum(positive.values())
        if total > 0:
            return {key: val / total for key, val in positive.items()}

        clean_fallback = [key for key in fallback_keys if key]
        if not clean_fallback:
            return {"product": 1.0}

        uniform = 1.0 / len(clean_fallback)
        return {key: uniform for key in clean_fallback}

    def _derive_selectivity_profile(
        self,
        metadata: Optional[ReactionMetadata],
        conditions: ReactionConditions,
        product_labels: List[str]
    ) -> Dict[str, float]:
        """Derive product selectivity profile from metadata and catalyst settings."""
        profile: Dict[str, Any] = {}
        fallback_keys = list(product_labels) if product_labels else ["product"]

        if metadata:
            matched_profile: Optional[Dict[str, float]] = None
            if conditions.solvent:
                for effect in metadata.solvent_effects:
                    if effect.matches(conditions.solvent) and effect.selectivity_profile:
                        matched_profile = effect.selectivity_profile
                        break
            if matched_profile is None:
                for effect in metadata.solvent_effects:
                    if effect.selectivity_profile:
                        matched_profile = effect.selectivity_profile
                        break
            if matched_profile:
                profile.update(matched_profile)
                fallback_keys = list(matched_profile.keys())

        catalyst_profile = conditions.catalyst.selectivity if conditions.catalyst else None
        if catalyst_profile:
            if profile:
                combined_keys = set(profile.keys()) | set(catalyst_profile.keys())
                combined: Dict[str, float] = {}
                for key in combined_keys:
                    base_val = profile.get(key, 1.0)
                    cat_val = catalyst_profile.get(key, 1.0)
                    try:
                        combined[key] = float(base_val) * float(cat_val)
                    except (TypeError, ValueError):
                        combined[key] = 0.0
                profile = combined
            else:
                profile = {key: float(val) for key, val in catalyst_profile.items()}
                fallback_keys = list(profile.keys())

        return self._normalize_distribution(profile, fallback_keys)

    def _selectivity_profile(
        self,
        path: ReactionPath,
        conditions: ReactionConditions,
        metadata: Optional[ReactionMetadata]
    ) -> tuple[List[str], Dict[str, float]]:
        """Resolve labels and selectivity distribution for current conditions."""
        product_labels = self._resolve_product_labels(metadata, path)
        profile = self._derive_selectivity_profile(metadata, conditions, product_labels)
        self._last_selectivity_profile = profile
        return product_labels, profile

    def calculate_activation_energy(
        self,
        reactants: List[Molecule],
        transition_state: TransitionState
    ) -> float:
        """Calculate activation energy (Ea) from reactant and TS energies."""
        reactant_energy = sum(m.energy for m in reactants)
        ea = transition_state.energy - reactant_energy
        return ea

    def arrhenius_rate_constant(
        self,
        activation_energy: float,
        temperature: float,
        pre_exponential_factor: float = 1e13,
        temperature_exponent: float = 0.0
    ) -> float:
        """
        Calculate rate constant using Arrhenius equation.

        k = A * exp(-Ea / (R*T))
        """
        k = pre_exponential_factor * (temperature ** temperature_exponent) * np.exp(
            -activation_energy / (self.R * temperature)
        )
        return k

    def eyring_rate_constant(
        self,
        delta_g_activation: float,
        temperature: float
    ) -> float:
        """
        Calculate rate constant using Eyring transition state theory.

        k = (k_B*T/h) * exp(-ΔG‡ / (R*T))
        """
        k_B_T_over_h = (self.k_B * temperature) / self.h
        # Convert to kcal/mol units
        k_B_T_over_h *= 2.390057e-4  # Conversion to kcal/mol
        k = k_B_T_over_h * np.exp(-delta_g_activation / (self.R * temperature))
        return k

    def calculate_gibbs_activation(
        self,
        activation_energy: float,
        activation_entropy: float,
        temperature: float
    ) -> float:
        """Calculate Gibbs free energy of activation."""
        # ΔG‡ = ΔH‡ - T*ΔS‡ (approximating ΔH‡ ≈ Ea)
        delta_g = activation_energy - temperature * activation_entropy / 1000.0  # Convert cal to kcal
        return delta_g

    def calculate_equilibrium_constant(
        self,
        delta_g_reaction: float,
        temperature: float
    ) -> float:
        """
        Calculate equilibrium constant from reaction free energy.

        K_eq = exp(-ΔG / (R*T))
        """
        k_eq = np.exp(-delta_g_reaction / (self.R * temperature))
        return k_eq

    def nudged_elastic_band(
        self,
        reactants: List[Molecule],
        products: List[Molecule],
        n_images: int = 10,
        spring_constant: float = 1.0
    ) -> ReactionPath:
        """
        Find reaction pathway using Nudged Elastic Band (NEB) method.

        This is a simplified implementation. Full NEB requires:
        1. Initial guess path (linear interpolation)
        2. Optimization of intermediate images
        3. Force projection (perpendicular to path)
        4. Spring forces between images
        """
        # Simplified: Linear interpolation between reactants and products
        reactant_energy = sum(m.energy for m in reactants)
        product_energy = sum(m.energy for m in products)

        # Estimate barrier height (rough approximation)
        delta_e = product_energy - reactant_energy
        barrier_estimate = max(15.0, abs(delta_e) * 0.3 + 12.0)  # Empirical but conservative

        # Create transition state
        ts_energy = reactant_energy + barrier_estimate
        ts = TransitionState(
            geometry=np.zeros((10, 3)),  # Placeholder
            energy=ts_energy,
            imaginary_frequency=-500.0,  # cm^-1
            reaction_coordinate=np.linspace(0, 1, n_images)
        )

        # Calculate thermodynamics
        delta_h = delta_e  # Approximation
        delta_s = sum(m.entropy for m in products) - sum(m.entropy for m in reactants)
        delta_g = delta_h - 298.15 * delta_s / 1000.0

        return ReactionPath(
            reactants=reactants,
            products=products,
            transition_states=[ts],
            intermediates=[],
            barriers_forward=[barrier_estimate],
            barriers_reverse=[barrier_estimate - delta_e],
            reaction_energy=delta_e,
            reaction_enthalpy=delta_h,
            reaction_entropy=delta_s,
            reaction_gibbs=delta_g
        )

    def apply_catalyst_effect(
        self,
        barrier: float,
        catalyst: Catalyst
    ) -> float:
        """Apply catalyst effect to reaction barrier."""
        return barrier - catalyst.barrier_reduction

    def simulate_reaction_kinetics(
        self,
        path: ReactionPath,
        conditions: ReactionConditions,
        initial_concentration: Optional[float] = None,
        time_points: Optional[np.ndarray] = None,
        reaction_name: Optional[str] = None,
        metadata: Optional[ReactionMetadata] = None,
        *,
        return_profiles: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]
    ]:
        """
        Simulate reaction kinetics over time.

        Returns:
            time, [reactant], [product] concentration profiles. If return_profiles=True,
            an additional dictionary of per-product concentration curves is returned.
        """
        if time_points is None:
            time_points = np.logspace(-6, 3, 100)  # 1 μs to 1000 s

        info = metadata or self.get_reaction_metadata(reaction_name)
        order = info.kinetics.reaction_order if info else 1
        initial_conc = initial_concentration if initial_concentration is not None else self.default_initial_concentration

        if info:
            ea = info.kinetics.activation_energy_kcal_per_mol
            pre_exp = info.kinetics.arrhenius_A
            temp_exp = info.kinetics.temperature_exponent
        else:
            ea = path.barriers_forward[0]
            pre_exp = 1e13
            temp_exp = 0.0

        # Apply catalyst if present
        if conditions.catalyst:
            ea = self.apply_catalyst_effect(ea, conditions.catalyst)

        # Calculate rate constant
        k = self.arrhenius_rate_constant(ea, conditions.temperature, pre_exp, temp_exp)
        k *= self._solvent_rate_factor(info, conditions.solvent)

        if order == 2:
            denominator = 1 + k * initial_conc * time_points
            reactant_conc = initial_conc / denominator
            product_conc = initial_conc - reactant_conc
        else:  # Treat all other orders as pseudo-first-order
            reactant_conc = initial_conc * np.exp(-k * time_points)
            product_conc = initial_conc - reactant_conc

        _, selectivity_profile = self._selectivity_profile(path, conditions, info)
        product_profiles: Dict[str, np.ndarray] = {
            key: product_conc * fraction
            for key, fraction in selectivity_profile.items()
        }
        product_profiles["total"] = product_conc
        self._last_product_profiles = product_profiles

        if return_profiles:
            return time_points, reactant_conc, product_conc, product_profiles
        return time_points, reactant_conc, product_conc

    def predict_reaction_kinetics(
        self,
        path: ReactionPath,
        conditions: ReactionConditions,
        reaction_name: Optional[str] = None,
        metadata: Optional[ReactionMetadata] = None,
        initial_concentration: Optional[float] = None
    ) -> ReactionKinetics:
        """Predict reaction kinetics parameters."""
        info = metadata or self.get_reaction_metadata(reaction_name)
        order = info.kinetics.reaction_order if info else 1
        initial_conc = initial_concentration if initial_concentration is not None else self.default_initial_concentration

        if info:
            ea = info.kinetics.activation_energy_kcal_per_mol
            pre_exp = info.kinetics.arrhenius_A
            temp_exp = info.kinetics.temperature_exponent
        else:
            ea = path.barriers_forward[0]
            pre_exp = 1e13
            temp_exp = 0.0

        # Apply catalyst
        if conditions.catalyst:
            ea = self.apply_catalyst_effect(ea, conditions.catalyst)

        # Calculate rate constant at given temperature
        k = self.arrhenius_rate_constant(ea, conditions.temperature, pre_exp, temp_exp)
        k *= self._solvent_rate_factor(info, conditions.solvent)

        half_life = self._compute_half_life(k, order, initial_conc)

        if info:
            info.thermodynamics.ensure_delta_g(conditions.temperature)
            k_eq = info.thermodynamics.equilibrium_constant(conditions.temperature, self.R)
        else:
            k_eq = self.calculate_equilibrium_constant(path.reaction_gibbs, conditions.temperature)

        _, selectivity_profile = self._selectivity_profile(path, conditions, info)

        return ReactionKinetics(
            rate_constant=k,
            activation_energy=ea,
            pre_exponential_factor=pre_exp,
            reaction_order=order,
            half_life=half_life,
            equilibrium_constant=k_eq,
            product_selectivity=selectivity_profile
        )

    def calculate_reaction_barrier(
        self,
        reactant_energy: float,
        product_energy: float,
        reaction_type: ReactionType = ReactionType.SYNTHESIS
    ) -> Tuple[float, float]:
        """
        Estimate reaction barriers using empirical relationships.

        Returns:
            forward_barrier, reverse_barrier (kcal/mol)
        """
        delta_e = product_energy - reactant_energy

        # Empirical barrier estimates based on reaction type
        if reaction_type == ReactionType.SYNTHESIS:
            forward_barrier = 15.0 + 0.3 * abs(delta_e)
        elif reaction_type == ReactionType.DECOMPOSITION:
            forward_barrier = 25.0 + 0.4 * abs(delta_e)
        elif reaction_type == ReactionType.SUBSTITUTION:
            forward_barrier = 12.0 + 0.25 * abs(delta_e)
        elif reaction_type == ReactionType.ADDITION:
            forward_barrier = 18.0 + 0.35 * abs(delta_e)
        elif reaction_type == ReactionType.ELIMINATION:
            forward_barrier = 22.0 + 0.4 * abs(delta_e)
        else:
            forward_barrier = 20.0 + 0.3 * abs(delta_e)

        reverse_barrier = forward_barrier - delta_e

        return forward_barrier, reverse_barrier

    def _solvent_rate_factor(
        self,
        metadata: Optional[ReactionMetadata],
        solvent: Optional[str]
    ) -> float:
        """Get multiplicative rate adjustment for solvent."""
        if metadata is None or not solvent:
            return 1.0

        for effect in metadata.solvent_effects:
            if effect.matches(solvent):
                return effect.rate_factor

        return 1.0

    def _compute_half_life(
        self,
        rate_constant: float,
        reaction_order: int,
        initial_concentration: float
    ) -> float:
        """Compute half-life based on reaction order."""
        if rate_constant <= 0:
            return np.inf

        if reaction_order == 1:
            return np.log(2) / rate_constant
        if reaction_order == 2:
            return 1.0 / (rate_constant * initial_concentration)

        # Fallback: treat as pseudo-first-order
        return np.log(2) / rate_constant

    def evaluate_safety(
        self,
        metadata: Optional[ReactionMetadata],
        conditions: ReactionConditions
    ) -> Dict[str, Any]:
        """Generate safety guidance and warnings for given conditions."""
        if metadata is None:
            return {"hazards": [], "ppe": [], "warnings": []}

        warnings: List[str] = []
        safety = metadata.safety
        temp_c = conditions.temperature - 273.15

        min_temp, max_temp = safety.temperature_limits_c
        if min_temp is not None and temp_c < min_temp:
            warnings.append(f"Temperature {temp_c:.1f} °C below recommended minimum {min_temp} °C.")
        if max_temp is not None and temp_c > max_temp:
            warnings.append(f"Temperature {temp_c:.1f} °C exceeds recommended maximum {max_temp} °C.")

        min_press, max_press = safety.pressure_limits_bar
        if min_press is not None and conditions.pressure < min_press:
            warnings.append(f"Pressure {conditions.pressure:.2f} bar below recommended minimum {min_press} bar.")
        if max_press is not None and conditions.pressure > max_press:
            warnings.append(f"Pressure {conditions.pressure:.2f} bar exceeds recommended maximum {max_press} bar.")

        if safety.flash_point_c is not None and temp_c > safety.flash_point_c:
            warnings.append(
                f"Operating temperature {temp_c:.1f} °C is above flash point ({safety.flash_point_c} °C); ensure inert atmosphere."
            )

        if conditions.solvent:
            warnings.append(f"Solvent: {conditions.solvent}. Verify compatibility with listed hazards.")

        return {
            "hazards": list(safety.hazards),
            "ppe": list(safety.ppe),
            "warnings": warnings,
        }

    def build_integration_payload(
        self,
        metadata: ReactionMetadata,
        kinetics: ReactionKinetics,
        conditions: ReactionConditions,
        initial_concentration: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create integration payload for materials and environmental modules."""
        initial_conc = initial_concentration if initial_concentration is not None else self.default_initial_concentration

        combined_hazards = {hazard.lower() for hazard in metadata.safety.hazards}
        for byproduct in metadata.byproducts:
            combined_hazards.update(h.lower() for h in byproduct.hazards)

        materials_payload = {
            "reaction_name": metadata.name,
            "links": metadata.materials.as_dict(),
            "kinetics": asdict(kinetics),
            "thermodynamics": metadata.thermodynamics.as_dict(),
            "selectivity": [
                {
                    "solvent": effect.solvent,
                    "selectivity_shift": effect.selectivity_shift,
                    "selectivity_profile": dict(effect.selectivity_profile),
                }
                for effect in metadata.solvent_effects
            ],
            "effects": metadata.material_effects,
            "hazards": sorted(combined_hazards),
        }

        environment_payload = []
        environment_effect_map: Dict[str, Dict[str, Any]] = {}
        for effect in metadata.environment_effects:
            key = effect.get("material_id") or effect.get("name")
            if not key:
                continue
            environment_effect_map[key.lower()] = dict(effect)

        default_target_material = next(iter(metadata.materials.product_ids or []), None)
        consumed_effects: set[str] = set()

        for byproduct in metadata.byproducts:
            phase = "gas" if "vapor" in byproduct.material_id.lower() or "oxide" in byproduct.material_id.lower() else "liquid"
            key = (byproduct.material_id or byproduct.name).lower()
            effect_metadata = environment_effect_map.get(key, {})
            if effect_metadata:
                consumed_effects.add(key)

            hazard_set = {h.lower() for h in byproduct.hazards}
            emission_duration = effect_metadata.get("exposure_hours", 1.0)

            half_life = effect_metadata.get("decay_half_life_hours")
            if half_life is None:
                if "corrosive" in hazard_set:
                    half_life = 6.0
                elif "toxic" in hazard_set:
                    half_life = 8.0
                else:
                    half_life = 24.0

            removal_eff = effect_metadata.get("removal_efficiency")
            if removal_eff is None:
                if byproduct.disposal == "scrubber":
                    removal_eff = 0.55
                elif byproduct.disposal in {"vented_condense", "burn_off"}:
                    removal_eff = 0.35
                else:
                    removal_eff = 0.1

            corrosion_multiplier = effect_metadata.get("corrosion_rate_multiplier")
            if corrosion_multiplier is None:
                corrosion_multiplier = 1.35 if "corrosive" in hazard_set else 1.05 if "toxic" in hazard_set else 1.0

            target_material = effect_metadata.get("target_material", default_target_material)
            baseline_corrosion = effect_metadata.get("baseline_corrosion_rate_mm_per_year")
            if baseline_corrosion is None and corrosion_multiplier > 1.0:
                baseline_corrosion = 0.2

            payload_entry = {
                "material_id": byproduct.material_id,
                "name": byproduct.name,
                "estimated_release_rate": kinetics.rate_constant * byproduct.yield_fraction * initial_conc,
                "hazards": list(byproduct.hazards),
                "disposal": byproduct.disposal,
                "phase": phase,
                "exposure_hours": emission_duration,
                "decay_half_life_hours": half_life,
                "removal_efficiency": removal_eff,
                "corrosion_rate_multiplier": corrosion_multiplier,
                "target_material": target_material,
                "baseline_corrosion_rate_mm_per_year": baseline_corrosion,
            }
            for key_name, value in effect_metadata.items():
                if key_name not in {
                    "material_id",
                    "name",
                    "decay_half_life_hours",
                    "removal_efficiency",
                    "corrosion_rate_multiplier",
                    "target_material",
                    "baseline_corrosion_rate_mm_per_year",
                    "exposure_hours",
                }:
                    payload_entry[key_name] = value

            environment_payload.append(payload_entry)

        for key, effect in environment_effect_map.items():
            if key in consumed_effects:
                continue
            environment_payload.append(dict(effect))

        return {
            "materials": materials_payload,
            "environment": environment_payload,
            "safety": self.evaluate_safety(metadata, conditions),
        }

    def analyze_catalysis_mechanism(
        self,
        reaction_name: str,
        catalyst: Catalyst
    ) -> Dict:
        """Analyze how catalyst affects reaction mechanism."""
        analysis = {
            "catalyst": catalyst.name,
            "mechanism": "heterogeneous" if "surface" in catalyst.name.lower() else "homogeneous",
            "barrier_reduction": catalyst.barrier_reduction,
            "selectivity": catalyst.selectivity,
            "active_sites": catalyst.active_sites,
            "rate_enhancement": np.exp(catalyst.barrier_reduction / (self.R * 298.15))
        }

        return analysis

    def multi_step_reaction_pathway(
        self,
        reactants: List[Molecule],
        products: List[Molecule],
        n_steps: int = 2
    ) -> List[ReactionPath]:
        """
        Generate multi-step reaction pathway with intermediates.
        """
        pathways = []

        # Energy spacing for intermediates
        reactant_energy = sum(m.energy for m in reactants)
        product_energy = sum(m.energy for m in products)
        energy_step = (product_energy - reactant_energy) / n_steps

        current_reactants = reactants

        for step in range(n_steps):
            # Create intermediate
            intermediate_energy = reactant_energy + (step + 1) * energy_step
            intermediate = Molecule(
                formula=f"I{step+1}",
                smiles="",
                energy=intermediate_energy,
                enthalpy=intermediate_energy,
                entropy=50.0
            )

            # Generate step
            if step < n_steps - 1:
                step_products = [intermediate]
            else:
                step_products = products

            path = self.nudged_elastic_band(current_reactants, step_products)
            pathways.append(path)

            current_reactants = step_products

        return pathways


def create_example_molecules() -> Tuple[List[Molecule], List[Molecule]]:
    """Create example reactants and products for Diels-Alder reaction."""
    # Butadiene (diene)
    diene = Molecule(
        formula="C4H6",
        smiles="C=CC=C",
        energy=0.0,
        enthalpy=0.0,
        entropy=60.0
    )

    # Ethylene (dienophile)
    dienophile = Molecule(
        formula="C2H4",
        smiles="C=C",
        energy=0.0,
        enthalpy=0.0,
        entropy=50.0
    )

    # Cyclohexene (product)
    product = Molecule(
        formula="C6H10",
        smiles="C1CC=CCC1",
        energy=-40.0,  # Exothermic
        enthalpy=-40.0,
        entropy=75.0
    )

    return [diene, dienophile], [product]


if __name__ == "__main__":
    print("Reaction Simulator Test\n")

    # Create simulator
    sim = ReactionSimulator()

    # Example: Diels-Alder reaction
    print("=== Diels-Alder Reaction ===")
    reactants, products = create_example_molecules()

    print("\nReactants:")
    for r in reactants:
        print(f"  {r.formula}: {r.energy:.2f} kcal/mol")

    print("\nProducts:")
    for p in products:
        print(f"  {p.formula}: {p.energy:.2f} kcal/mol")

    # Find reaction pathway
    print("\nFinding reaction pathway (NEB)...")
    path = sim.nudged_elastic_band(reactants, products)

    print(f"\nReaction Energy (ΔE): {path.reaction_energy:.2f} kcal/mol")
    print(f"Reaction Enthalpy (ΔH): {path.reaction_enthalpy:.2f} kcal/mol")
    print(f"Reaction Entropy (ΔS): {path.reaction_entropy:.2f} cal/(mol*K)")
    print(f"Reaction Gibbs (ΔG @ 298K): {path.reaction_gibbs:.2f} kcal/mol")
    print(f"\nForward Barrier: {path.barriers_forward[0]:.2f} kcal/mol")
    print(f"Reverse Barrier: {path.barriers_reverse[0]:.2f} kcal/mol")

    # Reaction conditions
    conditions = ReactionConditions(
        temperature=298.15,  # 25°C
        pressure=1.0,
        solvent="toluene"
    )

    # Calculate kinetics
    print("\n=== Reaction Kinetics ===")
    kinetics = sim.predict_reaction_kinetics(path, conditions)

    print(f"Rate constant (k): {kinetics.rate_constant:.2e} s^-1")
    print(f"Half-life: {kinetics.half_life:.2e} s ({kinetics.half_life/3600:.2f} hours)")
    print(f"Equilibrium constant: {kinetics.equilibrium_constant:.2e}")
    if kinetics.product_selectivity:
        print(f"Selectivity profile: {kinetics.product_selectivity}")

    # Simulate concentration profiles
    time, reactant_conc, product_conc, product_profiles = sim.simulate_reaction_kinetics(
        path, conditions, initial_concentration=1.0, return_profiles=True
    )

    print(f"\nAt t = 1 hour:")
    t_1h = np.argmin(np.abs(time - 3600))
    print(f"  [Reactant]: {reactant_conc[t_1h]:.4f} M")
    print(f"  [Product]: {product_conc[t_1h]:.4f} M")
    for name, profile in product_profiles.items():
        if name == "total":
            continue
        print(f"  [{name} channel]: {profile[t_1h]:.4f} M")
    print(f"  Conversion: {(1 - reactant_conc[t_1h])*100:.1f}%")

    # Test with catalyst
    print("\n=== With Lewis Acid Catalyst ===")
    catalyst = Catalyst(
        name="AlCl3",
        formula="AlCl3",
        active_sites=["Al"],
        barrier_reduction=5.0,  # Lowers barrier by 5 kcal/mol
        selectivity={"endo": 0.8, "exo": 0.2}
    )

    conditions_cat = ReactionConditions(
        temperature=298.15,
        pressure=1.0,
        solvent="dichloromethane",
        catalyst=catalyst
    )

    kinetics_cat = sim.predict_reaction_kinetics(path, conditions_cat)

    print(f"Rate constant (k): {kinetics_cat.rate_constant:.2e} s^-1")
    print(f"Half-life: {kinetics_cat.half_life:.2e} s ({kinetics_cat.half_life/3600:.2f} hours)")
    print(f"Rate enhancement: {kinetics_cat.rate_constant / kinetics.rate_constant:.1f}x")
    if kinetics_cat.product_selectivity:
        print(f"Selectivity profile (catalyst): {kinetics_cat.product_selectivity}")

    # Analyze catalysis
    cat_analysis = sim.analyze_catalysis_mechanism("diels_alder", catalyst)
    print(f"\nCatalysis mechanism: {cat_analysis['mechanism']}")
    print(f"Theoretical rate enhancement: {cat_analysis['rate_enhancement']:.1f}x")
    print(f"Selectivity: {cat_analysis['selectivity']}")

    print("\nReaction Simulator ready!")
