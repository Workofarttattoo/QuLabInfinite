"""
Comprehensive Quantum Teleportation Protocols.

Implements all known quantum teleportation variants for the discovery framework:
- Standard Bell-state teleportation (1 qubit)
- Entanglement swapping (extend range)
- Quantum repeater chains (long-distance)
- Multi-qubit teleportation
- Distributed teleportation
- Teleportation with error correction

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

from typing import Tuple, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOL TYPES
# ═══════════════════════════════════════════════════════════════════════════

class TeleportationProtocolType(Enum):
    """Types of quantum teleportation protocols."""
    BELL_STATE = "bell_state"                    # Standard Bennett et al. 1993
    ENTANGLEMENT_SWAPPING = "entanglement_swapping"  # Extend range
    QUANTUM_REPEATER = "quantum_repeater"        # Multiple hops
    LONG_DISTANCE = "long_distance"              # With repeater chains
    MULTI_QUBIT = "multi_qubit"                  # Teleport multiple qubits
    DISTRIBUTED = "distributed"                  # Distributed across nodes
    ERROR_CORRECTED = "error_corrected"          # With error correction


@dataclass
class ProtocolParameters:
    """Parameters defining a teleportation protocol."""
    protocol_type: TeleportationProtocolType
    num_qubits: int = 1                          # Qubits being teleported
    distance_km: float = 1.0                     # Communication distance
    bell_pair_fidelity: float = 0.99             # Quality of entanglement
    measurement_fidelity: float = 0.99           # Quality of measurement
    gate_fidelity: float = 0.99                  # Quality of gates
    classical_error_prob: float = 0.0            # Classical bit flip probability
    decoherence_time_us: float = 100.0           # Coherence time (microseconds)
    operation_time_us: float = 1.0               # Time to perform protocol
    error_correction_enabled: bool = False       # Use error correction?
    num_repeaters: int = 0                       # Number of repeater stations


@dataclass
class ProtocolResult:
    """Result of executing a teleportation protocol."""
    protocol_type: TeleportationProtocolType
    fidelity: float                              # State fidelity achieved
    success_probability: float                   # P(successful teleportation)
    classical_bits_needed: int                   # Classical bits communicated
    quantum_resources_needed: int                # Qubits needed
    time_required_us: float                      # Microseconds
    error_sources: Dict[str, float]              # Breakdown of errors
    scaling_factor: float = 1.0                  # Relative to standard teleportation
    optimal_for_distance: bool = False           # Best choice at this distance?
    required_gate_fidelity: float = 0.99         # Minimum gate fidelity needed


# ═══════════════════════════════════════════════════════════════════════════
# BASE PROTOCOL CLASS
# ═══════════════════════════════════════════════════════════════════════════

class TeleportationProtocol(ABC):
    """Abstract base class for teleportation protocols."""

    def __init__(self, params: ProtocolParameters):
        """Initialize protocol with parameters."""
        self.params = params
        self.validate_parameters()

    def validate_parameters(self):
        """Validate that parameters are physically reasonable."""
        if not (0 <= self.params.bell_pair_fidelity <= 1.0):
            raise ValueError("Bell pair fidelity must be in [0, 1]")
        if not (0 <= self.params.measurement_fidelity <= 1.0):
            raise ValueError("Measurement fidelity must be in [0, 1]")
        if not (0 <= self.params.gate_fidelity <= 1.0):
            raise ValueError("Gate fidelity must be in [0, 1]")
        if self.params.num_qubits < 1:
            raise ValueError("Must teleport at least 1 qubit")

    @abstractmethod
    def execute(self) -> ProtocolResult:
        """Execute the protocol and return results."""
        pass

    def _calculate_fidelity(self) -> float:
        """Calculate overall fidelity from component fidelities."""
        # Product of individual components (cascade effect)
        return (
            self.params.bell_pair_fidelity *
            self.params.measurement_fidelity *
            (self.params.gate_fidelity ** (2 * self.params.num_qubits))  # Two gates per qubit
        )

    def _calculate_decoherence_loss(self) -> float:
        """Loss due to decoherence during protocol execution."""
        if self.params.decoherence_time_us == 0:
            return 0
        # Exponential decay: exp(-t/T2)
        decay_rate = self.params.operation_time_us / self.params.decoherence_time_us
        return 1.0 - np.exp(-decay_rate)


# ═══════════════════════════════════════════════════════════════════════════
# 1. STANDARD BELL STATE TELEPORTATION (Bennett et al. 1993)
# ═══════════════════════════════════════════════════════════════════════════

class BellStateTeleportation(TeleportationProtocol):
    """
    Standard quantum teleportation protocol.

    Steps:
    1. Alice and Bob share Bell pair |Φ⁺⟩
    2. Alice performs Bell measurement on qubit + her half of pair
    3. Alice sends 2 classical bits to Bob
    4. Bob applies correction based on measurement

    Resources:
    - 1 ebit (Einstein-Podolsky-Rosen pair)
    - 2 classical bits
    - Operations: 1 CNOT, 2 H gates, 2 measurements, 2 corrections
    """

    def execute(self) -> ProtocolResult:
        """Execute standard Bell state teleportation."""
        # Component fidelities
        base_fidelity = self._calculate_fidelity()
        decoherence_loss = self._calculate_decoherence_loss()

        # Classical bit errors (if any)
        classical_error_impact = 1.0
        if self.params.classical_error_prob > 0:
            # Each classical bit error applies wrong correction
            classical_error_impact = (1.0 - self.params.classical_error_prob) ** 2

        final_fidelity = base_fidelity * (1.0 - decoherence_loss) * classical_error_impact

        return ProtocolResult(
            protocol_type=TeleportationProtocolType.BELL_STATE,
            fidelity=final_fidelity,
            success_probability=final_fidelity,
            classical_bits_needed=2,
            quantum_resources_needed=3,  # Alice's qubit + Bell pair
            time_required_us=self.params.operation_time_us,
            error_sources={
                "bell_pair_error": 1.0 - self.params.bell_pair_fidelity,
                "measurement_error": 1.0 - self.params.measurement_fidelity,
                "gate_error": 1.0 - (self.params.gate_fidelity ** 4),
                "decoherence": decoherence_loss,
                "classical_bit_error": 1.0 - classical_error_impact,
            },
            scaling_factor=1.0,
            optimal_for_distance=True if self.params.distance_km < 10 else False,
            required_gate_fidelity=0.99,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. ENTANGLEMENT SWAPPING (Extend Range)
# ═══════════════════════════════════════════════════════════════════════════

class EntanglementSwapping(TeleportationProtocol):
    """
    Entanglement swapping protocol.

    Extends teleportation range by connecting two Bell pairs.

    Steps:
    1. Alice-Bob share Bell pair A
    2. Bob-Charlie share Bell pair B
    3. Bob performs Bell measurement on (A, B)
    4. Bob sends 2 classical bits to Alice and Charlie
    5. Result: Alice-Charlie entangled

    Advantage: Extends range without direct Alice-Charlie connection
    Disadvantage: Higher overhead, more measurements

    Resources:
    - 2 ebits (two Bell pairs)
    - 2 classical bits
    - Operations: 1 CNOT, 1 H, 2 measurements, corrections
    """

    def execute(self) -> ProtocolResult:
        """Execute entanglement swapping."""
        # More complex: involves two Bell measurements
        base_fidelity = self._calculate_fidelity()

        # Entanglement swapping adds extra measurement error
        # (need to measure Bell basis between two pairs)
        swapping_overhead = 1.0 - (1.0 - self.params.measurement_fidelity)

        decoherence_loss = self._calculate_decoherence_loss()

        final_fidelity = base_fidelity * swapping_overhead * (1.0 - decoherence_loss)

        return ProtocolResult(
            protocol_type=TeleportationProtocolType.ENTANGLEMENT_SWAPPING,
            fidelity=final_fidelity,
            success_probability=final_fidelity,
            classical_bits_needed=2,
            quantum_resources_needed=4,  # Two Bell pairs
            time_required_us=self.params.operation_time_us * 1.5,  # Extra measurement time
            error_sources={
                "first_bell_pair": 1.0 - self.params.bell_pair_fidelity,
                "second_bell_pair": 1.0 - self.params.bell_pair_fidelity,
                "swapping_measurement": 1.0 - self.params.measurement_fidelity,
                "decoherence": decoherence_loss,
            },
            scaling_factor=0.95,  # Slightly worse than direct
            optimal_for_distance=True if 10 < self.params.distance_km < 100 else False,
            required_gate_fidelity=0.985,  # More stringent
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. QUANTUM REPEATER CHAINS (Long Distance)
# ═══════════════════════════════════════════════════════════════════════════

class QuantumRepeaterChain(TeleportationProtocol):
    """
    Quantum repeater protocol for long-distance teleportation.

    Uses repeater stations to extend range:
    Alice → Repeater1 → Repeater2 → ... → Bob

    Each hop:
    1. Create local Bell pair
    2. Perform entanglement swapping
    3. Pass on to next repeater

    Advantage: Extends range exponentially with repeater number
    Disadvantage: High overhead, complex, many failure points

    Resources per repeater:
    - Bell pair generation and storage
    - Bell measurement capability
    - Classical feedback capability
    """

    def execute(self) -> ProtocolResult:
        """Execute quantum repeater protocol."""
        num_hops = self.params.num_repeaters + 1  # +1 for destination

        # Each hop introduces errors
        hop_fidelity = self.params.bell_pair_fidelity * self.params.measurement_fidelity

        # Cascade: (fidelity)^(num_hops)
        cascaded_fidelity = hop_fidelity ** num_hops

        # Gate overhead
        gates_per_hop = 2  # CNOT + H per measurement
        gate_overhead = self.params.gate_fidelity ** (gates_per_hop * num_hops)

        # Decoherence: distributed across hops
        # Each repeater needs to store qubit for some time
        decoherence_per_hop = self._calculate_decoherence_loss()
        total_decoherence = 1.0 - (1.0 - decoherence_per_hop) ** num_hops

        final_fidelity = cascaded_fidelity * gate_overhead * (1.0 - total_decoherence)

        return ProtocolResult(
            protocol_type=TeleportationProtocolType.QUANTUM_REPEATER,
            fidelity=final_fidelity,
            success_probability=final_fidelity,
            classical_bits_needed=2 * num_hops,  # 2 bits per hop
            quantum_resources_needed=3 * num_hops,  # 3 qubits per repeater
            time_required_us=self.params.operation_time_us * num_hops,
            error_sources={
                "cascaded_bell_pair_errors": 1.0 - hop_fidelity,
                "gate_errors": 1.0 - gate_overhead,
                "total_decoherence": total_decoherence,
                "classical_overhead": 0.01 * num_hops,
            },
            scaling_factor=0.8 if num_hops <= 5 else 0.5,  # Gets worse with more hops
            optimal_for_distance=self.params.distance_km > 100,
            required_gate_fidelity=0.995,  # Very stringent
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4. MULTI-QUBIT TELEPORTATION
# ═══════════════════════════════════════════════════════════════════════════

class MultiQubitTeleportation(TeleportationProtocol):
    """
    Teleport multiple qubits in one protocol.

    Either:
    - Sequential: Teleport qubits one by one (uses more time)
    - Parallel: Use entangled resource state (uses more qubits)

    Resources scale with number of qubits.
    """

    def execute(self) -> ProtocolResult:
        """Execute multi-qubit teleportation."""
        num_qubits = self.params.num_qubits

        # Fidelity degrades with number of qubits (more gates)
        gate_error_per_qubit = 1.0 - (self.params.gate_fidelity ** 4)  # 4 gates per qubit
        total_gate_error = 1.0 - (1.0 - gate_error_per_qubit) ** num_qubits

        base_fidelity = self._calculate_fidelity()
        multi_qubit_fidelity = base_fidelity * (1.0 - total_gate_error)

        return ProtocolResult(
            protocol_type=TeleportationProtocolType.MULTI_QUBIT,
            fidelity=multi_qubit_fidelity,
            success_probability=multi_qubit_fidelity,
            classical_bits_needed=2 * num_qubits,  # 2 bits per qubit
            quantum_resources_needed=3 * num_qubits,  # 3 qubits per state qubit
            time_required_us=self.params.operation_time_us * num_qubits,
            error_sources={
                "bell_pair_error": 1.0 - self.params.bell_pair_fidelity,
                "measurement_error": 1.0 - self.params.measurement_fidelity,
                "gate_error_cascade": total_gate_error,
            },
            scaling_factor=1.0 / (1.0 + 0.05 * num_qubits),  # Degrades with qubits
            optimal_for_distance=self.params.distance_km < 100,
            required_gate_fidelity=0.99 + (0.005 * num_qubits),  # Stricter for more qubits
        )


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOL FACTORY
# ═══════════════════════════════════════════════════════════════════════════

class ProtocolFactory:
    """Factory for creating appropriate teleportation protocols."""

    PROTOCOLS = {
        TeleportationProtocolType.BELL_STATE: BellStateTeleportation,
        TeleportationProtocolType.ENTANGLEMENT_SWAPPING: EntanglementSwapping,
        TeleportationProtocolType.QUANTUM_REPEATER: QuantumRepeaterChain,
        TeleportationProtocolType.MULTI_QUBIT: MultiQubitTeleportation,
    }

    @classmethod
    def create_protocol(
        cls,
        protocol_type: TeleportationProtocolType,
        params: ProtocolParameters
    ) -> TeleportationProtocol:
        """Create a protocol instance."""
        if protocol_type not in cls.PROTOCOLS:
            raise ValueError(f"Unknown protocol type: {protocol_type}")

        protocol_class = cls.PROTOCOLS[protocol_type]
        return protocol_class(params)

    @classmethod
    def create_optimal_protocol(
        cls,
        distance_km: float,
        num_qubits: int = 1,
        bell_pair_fidelity: float = 0.99,
        gate_fidelity: float = 0.99,
    ) -> Tuple[TeleportationProtocol, ProtocolResult]:
        """
        Select and create the optimal protocol for given constraints.

        Heuristic:
        - <10 km: Direct Bell state teleportation
        - 10-100 km: Entanglement swapping
        - >100 km: Quantum repeater chain
        """

        if distance_km < 10:
            protocol_type = TeleportationProtocolType.BELL_STATE
            repeaters = 0
        elif distance_km < 100:
            protocol_type = TeleportationProtocolType.ENTANGLEMENT_SWAPPING
            repeaters = 1
        else:
            protocol_type = TeleportationProtocolType.QUANTUM_REPEATER
            repeaters = max(2, int(np.log2(distance_km / 100)))  # Exponential scaling

        params = ProtocolParameters(
            protocol_type=protocol_type,
            num_qubits=num_qubits,
            distance_km=distance_km,
            bell_pair_fidelity=bell_pair_fidelity,
            gate_fidelity=gate_fidelity,
            num_repeaters=repeaters,
        )

        protocol = cls.create_protocol(protocol_type, params)
        result = protocol.execute()

        return protocol, result


# ═══════════════════════════════════════════════════════════════════════════
# PROTOCOL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def compare_protocols_at_distance(
    distance_km: float,
    bell_pair_fidelity: float = 0.99,
    gate_fidelity: float = 0.99,
) -> Dict[str, ProtocolResult]:
    """
    Compare all protocols at a given distance.

    Returns results for all applicable protocols.
    """
    results = {}

    # Bell state (short range)
    if distance_km < 50:
        params = ProtocolParameters(
            protocol_type=TeleportationProtocolType.BELL_STATE,
            distance_km=distance_km,
            bell_pair_fidelity=bell_pair_fidelity,
            gate_fidelity=gate_fidelity,
        )
        protocol = BellStateTeleportation(params)
        results["Bell State"] = protocol.execute()

    # Entanglement swapping
    if distance_km < 500:
        params = ProtocolParameters(
            protocol_type=TeleportationProtocolType.ENTANGLEMENT_SWAPPING,
            distance_km=distance_km,
            bell_pair_fidelity=bell_pair_fidelity,
            gate_fidelity=gate_fidelity,
            num_repeaters=1,
        )
        protocol = EntanglementSwapping(params)
        results["Entanglement Swapping"] = protocol.execute()

    # Quantum repeater (long range)
    repeaters = max(2, int(np.log2(max(1, distance_km / 100))))
    params = ProtocolParameters(
        protocol_type=TeleportationProtocolType.QUANTUM_REPEATER,
        distance_km=distance_km,
        bell_pair_fidelity=bell_pair_fidelity,
        gate_fidelity=gate_fidelity,
        num_repeaters=repeaters,
    )
    protocol = QuantumRepeaterChain(params)
    results["Quantum Repeater"] = protocol.execute()

    return results
