"""
Quantum Teleportation Scaling Studies.

Analyzes how quantum teleportation scales with:
- Distance (point-to-point, repeater chains)
- Number of qubits
- Gate fidelity requirements
- Error correction overhead
- Hardware resource requirements

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SCALING ANALYSIS RESULTS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DistanceScalingResult:
    """Results of analyzing distance scaling."""
    distance_km: float
    protocol_type: str
    required_gate_fidelity: float
    fidelity_achievable: float
    num_repeaters_needed: int
    repeater_spacing_km: float
    total_resources: int
    feasibility: str


@dataclass
class QubitScalingResult:
    """Results of analyzing qubit scaling."""
    num_qubits: int
    physical_qubits_needed: int
    logical_qubits_needed: int
    error_correction_ratio: float
    surface_code_distance: int
    coherence_time_requirement_ms: float
    gate_time_requirement_ns: float


@dataclass
class FidelityScalingResult:
    """Results of analyzing fidelity requirements."""
    target_fidelity: float
    required_component_fidelity: float
    gate_fidelity_needed: float
    measurement_fidelity_needed: float
    two_qubit_gate_fidelity_needed: float
    current_best_achieved: float
    gap_to_requirement: float


@dataclass
class ErrorBudgetResult:
    """Results of error budget analysis."""
    total_error_budget: float
    photon_loss_budget: float
    gate_error_budget: float
    measurement_error_budget: float
    decoherence_budget: float
    error_correction_overhead: float
    resources_per_error_corrected_qubit: int


# ═══════════════════════════════════════════════════════════════════════════
# SCALING ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class ScalingAnalyzer:
    """Analyzes scaling properties of quantum teleportation."""

    # Current hardware state (Oct 2025)
    CURRENT_STATE = {
        "best_gate_fidelity": 0.998,  # Google Willow
        "best_two_qubit_fidelity": 0.9975,  # Two-qubit gates
        "best_single_qubit_fidelity": 0.9999,  # IonQ
        "max_qubits": 433,  # IBM
        "best_coherence_time_us": 1000.0,  # Trapped ions
        "typical_gate_time_ns": 100.0,  # Superconducting qubits
    }

    @staticmethod
    def analyze_distance_scaling(
        distance_km: float,
        target_fidelity: float = 0.95,
        protocol: str = "optimal"
    ) -> DistanceScalingResult:
        """
        Analyze how quantum teleportation scales with distance.

        Rules:
        - < 10 km: Direct Bell state (no repeaters)
        - 10-100 km: Entanglement swapping (1-2 repeaters)
        - > 100 km: Quantum repeater chains

        Args:
            distance_km: Communication distance
            target_fidelity: Target output fidelity
            protocol: Protocol to use ("optimal", "bell", "swapping", "repeater")

        Returns:
            DistanceScalingResult with feasibility analysis
        """
        if protocol == "optimal":
            if distance_km < 10:
                protocol = "bell"
            elif distance_km < 100:
                protocol = "swapping"
            else:
                protocol = "repeater"

        # Calculate number of repeaters needed
        if distance_km < 10:
            num_repeaters = 0
            repeater_spacing = float('inf')
        elif distance_km < 100:
            num_repeaters = 1
            repeater_spacing = distance_km / 2
        else:
            # For long distance: spacing ~ 10-50 km depending on fidelity
            repeater_spacing = max(10.0, 100.0 * (0.99 ** 4))  # Heuristic
            num_repeaters = max(2, int(np.ceil(distance_km / repeater_spacing)))
            repeater_spacing = distance_km / num_repeaters

        # Required gate fidelity increases with distance (due to more gates)
        # F_target = (F_gate)^(4 + 4*num_repeaters)  (rough scaling)
        num_gates = 4 + (4 * num_repeaters)
        required_gate_fidelity = target_fidelity ** (1.0 / num_gates)

        # Current achievable fidelity
        current_best = ScalingAnalyzer.CURRENT_STATE["best_gate_fidelity"]
        fidelity_achievable = current_best ** num_gates

        # Total resources: 3 qubits per hop + error correction
        qubits_per_hop = 3
        total_resources = qubits_per_hop * (1 + num_repeaters)
        # Add error correction overhead
        total_resources = int(total_resources * 100)  # Surface code ~100:1 ratio

        # Determine feasibility
        if fidelity_achievable > target_fidelity:
            feasibility = "FEASIBLE_NOW"
        elif required_gate_fidelity < 0.99:
            feasibility = "NEAR_TERM (1-2 years)"
        elif required_gate_fidelity < 0.9975:
            feasibility = "MEDIUM_TERM (2-5 years)"
        else:
            feasibility = "LONG_TERM (5+ years)"

        return DistanceScalingResult(
            distance_km=distance_km,
            protocol_type=protocol,
            required_gate_fidelity=required_gate_fidelity,
            fidelity_achievable=fidelity_achievable,
            num_repeaters_needed=num_repeaters,
            repeater_spacing_km=repeater_spacing,
            total_resources=total_resources,
            feasibility=feasibility,
        )

    @staticmethod
    def analyze_qubit_scaling(
        num_qubits: int,
        use_error_correction: bool = True,
    ) -> QubitScalingResult:
        """
        Analyze resource requirements for teleporting N qubits.

        Error correction overhead (Surface codes):
        - Logical qubit requires 1000-10000 physical qubits depending on code distance
        - Code distance d requires ~2d² syndrome extraction

        Args:
            num_qubits: Number of qubits to teleport
            use_error_correction: Whether to include error correction

        Returns:
            QubitScalingResult with resource requirements
        """
        # Physical qubits without error correction
        # Standard teleportation: 3 qubits + 2 classical bits
        physical_qubits_base = 3 * num_qubits

        if not use_error_correction:
            return QubitScalingResult(
                num_qubits=num_qubits,
                physical_qubits_needed=physical_qubits_base,
                logical_qubits_needed=num_qubits,
                error_correction_ratio=1.0,
                surface_code_distance=1,
                coherence_time_requirement_ms=1.0,
                gate_time_requirement_ns=100.0,
            )

        # With error correction (surface code)
        # Assume target gate error ~ 10^-4
        # Required code distance: d = (log10(p) / log10(0.1)) where p is gate error
        target_gate_error = 1e-4
        current_gate_error = 1e-3

        # Code distance needed
        if current_gate_error < target_gate_error:
            code_distance = 1
        else:
            # Rough formula: error reduces exponentially with distance
            code_distance = max(3, int(np.log(current_gate_error / target_gate_error) / np.log(2)))

        # Physical qubits per logical qubit: ~2*(2d-1)² for surface codes
        qubits_per_logical = 2 * (2 * code_distance - 1) ** 2

        # Total physical qubits
        logical_qubits = num_qubits
        physical_qubits_total = logical_qubits * qubits_per_logical

        # Coherence time requirement: must maintain coherence for protocol duration
        # Protocol time ~ 1 µs, coherence time ~ 1-100 ms depending on qubits
        coherence_time_ms = max(1.0, num_qubits * 0.5)

        # Gate time requirement: tighter for more qubits
        gate_time_ns = max(50.0, 100.0 / (1 + np.sqrt(num_qubits)))

        return QubitScalingResult(
            num_qubits=num_qubits,
            physical_qubits_needed=int(physical_qubits_total),
            logical_qubits_needed=logical_qubits,
            error_correction_ratio=physical_qubits_total / max(1, physical_qubits_base),
            surface_code_distance=code_distance,
            coherence_time_requirement_ms=coherence_time_ms,
            gate_time_requirement_ns=gate_time_ns,
        )

    @staticmethod
    def analyze_fidelity_scaling(
        target_fidelity: float,
        num_qubits: int = 1,
        num_gates: int = 4,
    ) -> FidelityScalingResult:
        """
        Analyze fidelity requirements for quantum teleportation.

        Fidelity is a product of component fidelities:
        F_total = F_bell_pair × F_measurement × (F_gate)^(num_gates)

        Args:
            target_fidelity: Desired output state fidelity
            num_qubits: Number of qubits (scales gates)
            num_gates: Number of quantum gates in protocol

        Returns:
            FidelityScalingResult with component requirements
        """
        # Total gates scale with qubits
        total_gates = num_gates * num_qubits

        # Assume roughly equal contribution from each component
        # F_target ≈ F_component ^ 3
        component_fidelity = target_fidelity ** (1.0 / 3.0)

        # More refined: F_total = F_bell × F_meas × F_gate^gates
        # Assume F_bell ≈ F_meas
        # F_gate_required = (F_target / (F_bell × F_meas))^(1/gates)
        assumed_bell = 0.99
        assumed_meas = 0.99

        effective_fidelity = target_fidelity / (assumed_bell * assumed_meas)
        required_gate_fidelity = effective_fidelity ** (1.0 / total_gates)

        # Two-qubit gate fidelity is typically lower
        required_two_qubit = max(0.99, required_gate_fidelity)

        # Current best achieved
        current_best_two_qubit = ScalingAnalyzer.CURRENT_STATE["best_two_qubit_fidelity"]
        gap = required_two_qubit - current_best_two_qubit

        # Can we achieve this?
        achievable_fidelity = (current_best_two_qubit ** total_gates) * (assumed_bell * assumed_meas)

        return FidelityScalingResult(
            target_fidelity=target_fidelity,
            required_component_fidelity=component_fidelity,
            gate_fidelity_needed=required_gate_fidelity,
            measurement_fidelity_needed=assumed_meas,
            two_qubit_gate_fidelity_needed=required_two_qubit,
            current_best_achieved=current_best_two_qubit,
            gap_to_requirement=gap,
        )

    @staticmethod
    def analyze_error_budget(
        distance_km: float,
        num_qubits: int = 1,
        target_fidelity: float = 0.95,
    ) -> ErrorBudgetResult:
        """
        Allocate error budget across different error sources.

        Error budget allocation (typical):
        - Photon loss: 30%
        - Gate errors: 40%
        - Measurement errors: 20%
        - Decoherence: 10%

        Args:
            distance_km: Communication distance
            num_qubits: Number of qubits
            target_fidelity: Target output fidelity

        Returns:
            ErrorBudgetResult with detailed error budget
        """
        # Total error budget
        total_error_budget = 1.0 - target_fidelity

        # Allocate error budget
        photon_loss_budget = total_error_budget * 0.30
        gate_error_budget = total_error_budget * 0.40
        measurement_error_budget = total_error_budget * 0.20
        decoherence_budget = total_error_budget * 0.10

        # Determine error correction overhead
        # With surface codes, error correction overhead ~100:1 at code distance 5-10
        if target_fidelity < 0.90:
            ec_overhead = 1.0  # No overhead, not worth it
        elif target_fidelity < 0.95:
            ec_overhead = 10.0
        elif target_fidelity < 0.99:
            ec_overhead = 100.0
        else:
            ec_overhead = 1000.0

        # Resources per error-corrected qubit
        resources_per_ecq = int(10 * ec_overhead)

        return ErrorBudgetResult(
            total_error_budget=total_error_budget,
            photon_loss_budget=photon_loss_budget,
            gate_error_budget=gate_error_budget,
            measurement_error_budget=measurement_error_budget,
            decoherence_budget=decoherence_budget,
            error_correction_overhead=ec_overhead,
            resources_per_error_corrected_qubit=resources_per_ecq,
        )

    @staticmethod
    def generate_scaling_curve(
        parameter: str,
        min_val: float,
        max_val: float,
        num_points: int = 20,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Generate scaling curves for visualization.

        Args:
            parameter: "distance", "qubits", "fidelity", or "error_correction"
            min_val: Minimum parameter value
            max_val: Maximum parameter value
            num_points: Number of points to generate
            **kwargs: Additional parameters

        Returns:
            Dictionary with scaling data for plotting
        """
        results = {
            "x": [],
            "required_fidelity": [],
            "required_resources": [],
            "feasibility": [],
            "years_to_achieve": [],
        }

        if parameter == "distance":
            distances = np.logspace(np.log10(min_val), np.log10(max_val), num_points)
            for d in distances:
                result = ScalingAnalyzer.analyze_distance_scaling(d, target_fidelity=0.95)
                results["x"].append(d)
                results["required_fidelity"].append(result.required_gate_fidelity)
                results["required_resources"].append(result.total_resources)
                results["feasibility"].append(result.feasibility)

        elif parameter == "qubits":
            qubit_counts = np.linspace(int(min_val), int(max_val), num_points)
            for q in qubit_counts:
                result = ScalingAnalyzer.analyze_qubit_scaling(int(q))
                results["x"].append(q)
                results["required_fidelity"].append(1.0)  # N/A
                results["required_resources"].append(result.physical_qubits_needed)
                results["feasibility"].append("N/A")

        elif parameter == "fidelity":
            fidelities = np.linspace(min_val, max_val, num_points)
            for f in fidelities:
                result = ScalingAnalyzer.analyze_fidelity_scaling(f)
                results["x"].append(f)
                results["required_fidelity"].append(result.two_qubit_gate_fidelity_needed)
                results["required_resources"].append(int(result.gap_to_requirement * 1000))
                results["feasibility"].append("N/A")

        return results


# ═══════════════════════════════════════════════════════════════════════════
# SCALING SUITE
# ═══════════════════════════════════════════════════════════════════════════

class ScalingStudiesSuite:
    """High-level interface for quantum teleportation scaling studies."""

    @staticmethod
    def comprehensive_analysis(
        distance_km: float = 100.0,
        num_qubits: int = 1,
        target_fidelity: float = 0.95,
    ) -> Dict[str, any]:
        """
        Run comprehensive scaling analysis for given scenario.

        Args:
            distance_km: Communication distance
            num_qubits: Number of qubits to teleport
            target_fidelity: Target output fidelity

        Returns:
            Dictionary with all scaling analyses
        """
        return {
            "distance": ScalingAnalyzer.analyze_distance_scaling(distance_km, target_fidelity),
            "qubits": ScalingAnalyzer.analyze_qubit_scaling(num_qubits),
            "fidelity": ScalingAnalyzer.analyze_fidelity_scaling(target_fidelity, num_qubits),
            "error_budget": ScalingAnalyzer.analyze_error_budget(distance_km, num_qubits, target_fidelity),
        }

    @staticmethod
    def scaling_limits() -> Dict[str, Tuple[float, float, str]]:
        """
        Return current scaling limits based on hardware state.

        Returns:
            Dictionary with distance, qubit, and fidelity limits
        """
        current = ScalingAnalyzer.CURRENT_STATE

        return {
            "max_distance_feasible_km": (100.0, "Distance before repeaters required"),
            "max_qubits_with_ec": (int(current["max_qubits"] / 100), "Logical qubits with error correction"),
            "min_gate_fidelity_needed": (0.9875, "For 99% teleportation fidelity"),
            "max_distance_without_ec": (10.0, "Distance without error correction"),
            "timeline_years_to_1000km": (5.0, "Years to feasible 1000km teleportation"),
        }

    @staticmethod
    def roadmap_to_milestone(
        distance_km: float = 100.0,
        target_fidelity: float = 0.95,
    ) -> List[Dict[str, any]]:
        """
        Generate a roadmap to achieve a teleportation milestone.

        Args:
            distance_km: Target distance
            target_fidelity: Target fidelity

        Returns:
            List of milestones with timeline and requirements
        """
        milestones = []

        # Phase 1: Near term (1-2 years)
        milestones.append({
            "phase": "Phase 1: Proof of Concept (1-2 years)",
            "distance_km": 10.0,
            "required_fidelity": 0.90,
            "repeaters": 0,
            "qubits_needed": 50,
            "key_challenge": "Achieve 99.5% gate fidelity on 2-qubit gates",
            "hardware": "IBM Quantum / Google Willow class",
        })

        # Phase 2: Medium term (2-5 years)
        if distance_km > 50 or target_fidelity > 0.95:
            milestones.append({
                "phase": "Phase 2: Extended Range (2-5 years)",
                "distance_km": 50.0,
                "required_fidelity": 0.93,
                "repeaters": 1,
                "qubits_needed": 500,
                "key_challenge": "Entanglement swapping fidelity > 90%",
                "hardware": "Next-generation quantum repeater nodes",
            })

        # Phase 3: Long term (5+ years)
        if distance_km > 100 or target_fidelity > 0.96:
            milestones.append({
                "phase": "Phase 3: Continental Scale (5+ years)",
                "distance_km": 100.0,
                "required_fidelity": 0.95,
                "repeaters": 2,
                "qubits_needed": 2000,
                "key_challenge": "Quantum repeater network coordination",
                "hardware": "Distributed quantum network infrastructure",
            })

        # Phase 4: Long distance (10+ years)
        if distance_km > 1000:
            milestones.append({
                "phase": "Phase 4: Global Quantum Internet (10+ years)",
                "distance_km": 1000.0,
                "required_fidelity": 0.95,
                "repeaters": 10,
                "qubits_needed": 50000,
                "key_challenge": "Quantum routing and memory management",
                "hardware": "Global quantum internet backbone",
            })

        return milestones
