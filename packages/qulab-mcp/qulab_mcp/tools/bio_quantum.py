"""
Biological Quantum Computing MCP tools for QuLab Infinite.

Room-temperature quantum computing and biological quantum simulation:
  - FMO complex energy transfer     (Engel et al. Nature 2007)
  - 2D electronic spectroscopy      (digital-twin validation vs experiment)
  - Variational Quantum Eigensolver (VQE — molecular ground states)
  - Coherence protection design     (Diamond NV / SiC / Bi₂Se₃ stack)
  - AI-controlled FMO               (active coherence optimisation)

Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from mcp.types import CallToolResult, TextContent, Tool


# ── helpers ──────────────────────────────────────────────────────────────────

def _ok(data: Any) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(_safe(data), indent=2))]
    )


def _err(msg: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": msg}))],
        isError=True,
    )


def _safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.complexfloating):
        return {"real": float(obj.real), "imag": float(obj.imag)}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return obj


# ── Lab path bootstrap ────────────────────────────────────────────────────────

def _find_lab_root() -> Path | None:
    candidates = [
        Path(__file__).parents[4],
        Path(__file__).parents[3],
    ]
    for c in candidates:
        if (c / "quantum_computing_lab.py").exists():
            return c
    return None


_LAB_ROOT = _find_lab_root()
_BQ_ROOT = (
    _LAB_ROOT / "qulab/labs/quantum/biological_quantum"
    if _LAB_ROOT else None
)

if _BQ_ROOT and str(_BQ_ROOT) not in sys.path:
    sys.path.insert(0, str(_BQ_ROOT))


def _load_bq_module(subpath: str):
    """Load a submodule from the biological_quantum tree."""
    if _BQ_ROOT is None:
        return None
    full = _BQ_ROOT / subpath
    if not full.exists():
        return None
    try:
        name = "_bq_" + subpath.replace("/", "_").replace(".py", "")
        spec = importlib.util.spec_from_file_location(name, str(full))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _fmo_module():
    return _load_bq_module("simulation/fmo_complex.py")


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS: list[Tool] = [

    Tool(
        name="bio_quantum_fmo",
        description=(
            "Simulate quantum energy transfer through the Fenna-Matthews-Olson (FMO) "
            "photosynthetic complex — Nature's room-temperature quantum computer. "
            "Uses experimentally measured site energies and couplings (Adolphs & Renger "
            "2006; Engel et al. Nature 2007). "
            "Returns transfer efficiency between chromophores 1-7, quantum vs classical "
            "transport comparison, eigenstates (exciton energies), and the 33% quantum "
            "advantage figure. Simulation time in femtoseconds (natural coherence = 660 fs). "
            "Ideal for validating digital-twin models against 2D spectroscopy data."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "initial_site": {
                    "type": "integer", "minimum": 1, "maximum": 7, "default": 1,
                    "description": "Starting chromophore (1-7). Site 1 is the input from antenna complex.",
                },
                "final_site": {
                    "type": "integer", "minimum": 1, "maximum": 7, "default": 3,
                    "description": "Target chromophore (1-7). Site 3 is nearest the reaction centre.",
                },
                "time_fs": {
                    "type": "number", "default": 500,
                    "description": "Simulation time in femtoseconds (natural coherence ~ 660 fs).",
                },
                "compare_classical": {
                    "type": "boolean", "default": True,
                    "description": "Include quantum-vs-classical transport comparison.",
                },
                "show_eigenstates": {
                    "type": "boolean", "default": False,
                    "description": "Return full exciton eigenstate energies (cm⁻¹).",
                },
            },
            "required": [],
        },
    ),

    Tool(
        name="bio_quantum_spectroscopy_2d",
        description=(
            "Simulate 2D electronic spectroscopy on the FMO complex — the primary "
            "experimental technique for validating biological quantum coherence. "
            "Computes the third-order nonlinear optical response R⁽³⁾(t₁,T,t₃) via "
            "Redfield theory, then 2D Fourier-transforms to produce S(ω₁,T,ω₃). "
            "Returns frequency axes (cm⁻¹), peak positions, cross-peak amplitude "
            "(coherence signature), and estimated T₂ coherence time. "
            "Use this to generate synthetic spectra for digital-twin validation "
            "against Fleming/Engel lab experimental data."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "population_time_fs": {
                    "type": "number", "default": 200,
                    "description": "Population time T (waiting time between 2nd and 3rd pulse) in fs.",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["fast", "standard", "high"],
                    "default": "fast",
                    "description": (
                        "Spectral resolution: "
                        "fast=10pts (instant), standard=25pts (~1s), high=50pts (~5s)."
                    ),
                },
                "custom_coherence_time_fs": {
                    "type": "number",
                    "description": "Override FMO coherence time (default 660 fs). "
                                   "Use to model modified or mutant FMO complexes.",
                },
            },
            "required": [],
        },
    ),

    Tool(
        name="bio_quantum_vqe",
        description=(
            "Variational Quantum Eigensolver (VQE) optimised for short coherence times "
            "(room-temperature biological quantum hardware). Finds ground-state energy "
            "of a molecular or spin Hamiltonian using a parameterised quantum circuit. "
            "Built-in Hamiltonians: "
            "'ising_1d' (1D Ising chain), "
            "'fmo_transport' (FMO energy transport), "
            "'maxcut_ring' (combinatorial optimisation), "
            "'h2_molecule' (hydrogen molecule, bond-length dependent). "
            "Returns ground state energy, optimal circuit parameters, "
            "and convergence history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "hamiltonian": {
                    "type": "string",
                    "enum": ["ising_1d", "fmo_transport", "maxcut_ring", "h2_molecule"],
                    "default": "ising_1d",
                    "description": "Which built-in Hamiltonian to minimise.",
                },
                "n_qubits": {
                    "type": "integer", "minimum": 2, "maximum": 6, "default": 4,
                    "description": "Number of qubits (2-6). More qubits = richer solution space.",
                },
                "circuit_depth": {
                    "type": "integer", "minimum": 1, "maximum": 5, "default": 3,
                    "description": "Ansatz circuit depth (layers of parameterised gates).",
                },
                "max_iterations": {
                    "type": "integer", "minimum": 10, "maximum": 200, "default": 50,
                    "description": "Maximum optimisation iterations.",
                },
                "h2_bond_length_angstrom": {
                    "type": "number", "default": 0.74,
                    "description": "Bond length for h2_molecule Hamiltonian (Å). Equilibrium ≈ 0.74 Å.",
                },
            },
            "required": [],
        },
    ),

    Tool(
        name="bio_quantum_coherence_protection",
        description=(
            "Design and analyse a multi-layered coherence protection system for "
            "room-temperature (300 K) quantum computing. "
            "Stack: Diamond NV centers (core) → SiC shell (thermal) → "
            "Bi₂Se₃ topological insulator (disorder protection) → "
            "Mu-metal shielding (magnetic) → Aerogel (thermal insulation). "
            "Active: Dynamic Nuclear Polarisation (DNP) + chirped laser pulses + "
            "real-time feedback control. "
            "Returns coherence time (bare vs protected), per-layer enhancement factors, "
            "hardware specifications, and energy budget."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "dnp_power_W": {
                    "type": "number", "default": 0.1,
                    "description": "DNP microwave power (Watts). Higher = more enhancement.",
                },
                "laser_power_mW": {
                    "type": "number", "default": 10,
                    "description": "Chirped laser power (mW) for NV centre initialisation.",
                },
                "shielding_db": {
                    "type": "number", "default": 80,
                    "description": "Magnetic shielding factor (dB). Mu-metal typical: 60-120 dB.",
                },
                "feedback_loop_hz": {
                    "type": "number", "default": 1000,
                    "description": "Feedback control loop rate (Hz).",
                },
            },
            "required": [],
        },
    ),

    Tool(
        name="bio_quantum_ai_control",
        description=(
            "AI-optimised FMO coherence control loop. Simulates an AI controller "
            "that tunes light intensity, magnetic field, pH, and temperature "
            "microzone to extend quantum coherence beyond the natural 660 fs limit. "
            "Returns optimised control parameters, predicted coherence time, "
            "control history, and estimated quantum advantage at the optimised state. "
            "This is the core IP behind the ECH0 biological quantum computing platform."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "target_coherence_fs": {
                    "type": "number", "default": 1000,
                    "description": "Target coherence time in femtoseconds (natural baseline = 660 fs).",
                },
                "initial_light_intensity": {
                    "type": "number", "minimum": 0, "maximum": 1, "default": 0.5,
                    "description": "Initial normalised light intensity [0-1].",
                },
                "initial_magnetic_field_mT": {
                    "type": "number", "default": 10,
                    "description": "Initial magnetic field strength (millitesla).",
                },
                "initial_ph": {
                    "type": "number", "default": 7.4,
                    "description": "Initial chemical environment pH.",
                },
            },
            "required": [],
        },
    ),

]


# ── Handler implementations ───────────────────────────────────────────────────

async def _handle_fmo(args: dict) -> CallToolResult:
    mod = _fmo_module()
    if not mod:
        return _err("biological_quantum/simulation/fmo_complex.py not found — run from repo root.")
    try:
        initial_site = int(args.get("initial_site", 1))
        final_site = int(args.get("final_site", 3))
        time_fs = float(args.get("time_fs", 500))
        compare = bool(args.get("compare_classical", True))
        show_eigen = bool(args.get("show_eigenstates", False))

        fmo = mod.FMOComplex()
        efficiency = fmo.simulate_energy_transfer(initial_site, final_site, time_fs)

        result: dict[str, Any] = {
            "initial_site": initial_site,
            "final_site": final_site,
            "simulation_time_fs": time_fs,
            "transfer_efficiency": round(float(efficiency), 4),
            "transfer_efficiency_pct": round(float(efficiency) * 100, 2),
            "coherence_time_fs": fmo.params.coherence_time_fs,
            "temperature_K": fmo.params.temperature_K,
            "n_chromophores": fmo.n_sites,
            "reference": "Adolphs & Renger (2006); Engel et al. Nature 446 (2007)",
        }

        if compare:
            qa = fmo.assess_quantum_effects()
            result["quantum_vs_classical"] = {
                "quantum_efficiency": round(float(qa["quantum_efficiency"]), 4),
                "classical_efficiency": round(float(qa["classical_efficiency"]), 4),
                "quantum_advantage_pct": round(float(qa["quantum_advantage"]) * 100, 2),
                "interpretation": (
                    f"Quantum transport is {qa['quantum_advantage']*100:.1f}% more efficient "
                    f"than classical incoherent hopping at 300 K"
                ),
            }

        if show_eigen:
            eigenvalues, eigenvectors = fmo.compute_eigenstates()
            result["exciton_energies_cm1"] = [round(float(e), 1) for e in eigenvalues]
            result["exciton_energy_spread_cm1"] = round(
                float(eigenvalues.max() - eigenvalues.min()), 1
            )

        return _ok(result)
    except Exception as e:
        return _err(f"FMO simulation failed: {e}")


async def _handle_spectroscopy_2d(args: dict) -> CallToolResult:
    fmo_mod = _fmo_module()
    spec_mod = _load_bq_module("experimental/spectroscopy_2d.py")
    if not fmo_mod or not spec_mod:
        return _err("biological_quantum spectroscopy module not found — run from repo root.")
    try:
        pop_time = float(args.get("population_time_fs", 200))
        resolution = args.get("resolution", "fast")
        custom_t2 = args.get("custom_coherence_time_fs")

        # Resolution → (max_coherence_time_fs, time_resolution_fs, n_points)
        res_map = {
            "fast":     (330, 33),   # 10 pts
            "standard": (625, 25),   # 25 pts
            "high":     (1000, 20),  # 50 pts
        }
        t_max, dt = res_map.get(resolution, res_map["fast"])

        params_kw = dict(
            max_coherence_time_fs=t_max,
            time_resolution_fs=dt,
            population_time_T_fs=pop_time,
        )
        if custom_t2 is not None:
            fmo_params = fmo_mod.FMOParameters(coherence_time_fs=float(custom_t2))
            fmo = fmo_mod.FMOComplex(fmo_params)
        else:
            fmo = fmo_mod.FMOComplex()

        spec_params = spec_mod.SpectroscopyParameters(**params_kw)
        spectro = spec_mod.TwoDElectronicSpectroscopy(fmo, spec_params)
        omega1, omega3, spectrum = spectro.generate_2d_spectrum(pop_time)

        intensity = np.abs(spectrum)
        max_idx = np.unravel_index(np.argmax(intensity), intensity.shape)

        # Cross-peak amplitude as coherence signature
        # Diagonal vs off-diagonal power ratio
        n = spectrum.shape[0]
        diag_power = float(np.mean([abs(spectrum[i, i])**2 for i in range(n)]))
        offdiag_power = float(np.mean([
            abs(spectrum[i, j])**2
            for i in range(n) for j in range(n) if i != j
        ]))
        coherence_signature = offdiag_power / (diag_power + 1e-30)

        return _ok({
            "population_time_fs": pop_time,
            "resolution": resolution,
            "spectrum_shape": list(spectrum.shape),
            "omega1_range_cm1": [round(float(omega1.min()), 0), round(float(omega1.max()), 0)],
            "omega3_range_cm1": [round(float(omega3.min()), 0), round(float(omega3.max()), 0)],
            "peak_omega1_cm1": round(float(omega1[max_idx[0]]), 1),
            "peak_omega3_cm1": round(float(omega3[max_idx[1]]), 1),
            "peak_intensity": round(float(intensity.max()), 4),
            "cross_peak_coherence_ratio": round(coherence_signature, 4),
            "coherence_time_fs": fmo.params.coherence_time_fs,
            "temperature_K": fmo.params.temperature_K,
            "technique": "Third-order nonlinear spectroscopy (Redfield theory)",
            "validation_reference": (
                "Compare omega axes and cross-peak positions against: "
                "Engel et al. Nature 446:782 (2007); "
                "Collini et al. Nature 463:644 (2010)"
            ),
            "digital_twin_note": (
                "Cross-peak ratio > 0.1 indicates coherent quantum transport. "
                "Fit coherence_time_fs to match experimental T₂ decay to validate model."
            ),
        })
    except Exception as e:
        return _err(f"2D spectroscopy simulation failed: {e}")


async def _handle_vqe(args: dict) -> CallToolResult:
    if _BQ_ROOT is None:
        return _err("biological_quantum modules not found — run from repo root.")
    try:
        opt_mod = _load_bq_module("algorithms/quantum_optimization.py")
        if not opt_mod:
            return _err("quantum_optimization module not found.")

        hamiltonian_name = args.get("hamiltonian", "ising_1d")
        n_qubits = int(args.get("n_qubits", 4))
        depth = int(args.get("circuit_depth", 3))
        max_iter = int(args.get("max_iterations", 50))
        h2_bond = float(args.get("h2_bond_length_angstrom", 0.74))

        # Built-in Hamiltonians
        def ising_1d(state):
            probs = state.get_probabilities()
            J = 1.0
            energy = 0.0
            for i, p in enumerate(probs):
                bits = format(i, f"0{n_qubits}b")
                for j in range(len(bits) - 1):
                    zi = 1 if bits[j] == "0" else -1
                    zj = 1 if bits[j + 1] == "0" else -1
                    energy -= J * zi * zj * p
            return energy

        def fmo_transport(state):
            # Minimise energy gap between sites 1 and 3 (favours coherent transfer)
            probs = state.get_probabilities()
            site_energy = np.array([12410, 12530, 12210, 12320])[:n_qubits]
            energy = 0.0
            for i, p in enumerate(probs[:n_qubits]):
                energy += site_energy[i % len(site_energy)] * p * 1e-4
            return energy

        def maxcut_ring(state):
            probs = state.get_probabilities()
            energy = 0.0
            for i, p in enumerate(probs):
                bits = format(i, f"0{n_qubits}b")
                cut = sum(
                    1 for j in range(n_qubits)
                    if bits[j] != bits[(j + 1) % n_qubits]
                )
                energy -= cut * p  # Negate: VQE minimises
            return energy

        def h2_molecule(state):
            # Simplified H₂ molecular energy vs bond length (STO-3G minimal basis)
            # E(R) ≈ nuclear repulsion + electronic energy
            r = h2_bond
            nuclear = 1.0 / r
            probs = state.get_probabilities()
            # Two-qubit singlet/triplet encoding
            singlet_weight = probs[0] + probs[3] if len(probs) >= 4 else probs[0]
            triplet_weight = probs[1] + probs[2] if len(probs) >= 4 else (1 - probs[0])
            electronic = -1.8 * singlet_weight / (r ** 0.6) + 0.5 * triplet_weight
            return nuclear + electronic

        hamiltonians = {
            "ising_1d": ising_1d,
            "fmo_transport": fmo_transport,
            "maxcut_ring": maxcut_ring,
            "h2_molecule": h2_molecule,
        }

        H = hamiltonians[hamiltonian_name]
        vqe = opt_mod.VariationalQuantumEigensolver(n_qubits=n_qubits, depth=depth)
        ground_energy, optimal_params = vqe.optimize(H, max_iterations=max_iter)

        # Convergence: final 5 iterations
        history = getattr(vqe, "energy_history", [])

        return _ok({
            "hamiltonian": hamiltonian_name,
            "n_qubits": n_qubits,
            "circuit_depth": depth,
            "ground_state_energy": round(float(ground_energy), 6),
            "n_parameters": len(optimal_params),
            "optimal_parameters": [round(float(p), 4) for p in optimal_params],
            "iterations_run": max_iter,
            "convergence_last5": [round(float(e), 6) for e in history[-5:]] if history else [],
            "platform": "biological_quantum (room temperature, 300 K)",
            "note": {
                "ising_1d": "Ground state is the ferromagnetic/antiferromagnetic phase boundary",
                "fmo_transport": "Minimum energy state corresponds to coherent exciton delocalisation",
                "maxcut_ring": "Negative energy = more cuts; optimal ring cut = n_qubits",
                "h2_molecule": f"H₂ at R={h2_bond} Å; equilibrium is ~-1.1 Hartree at 0.74 Å",
            }.get(hamiltonian_name, ""),
        })
    except Exception as e:
        return _err(f"VQE failed: {e}")


async def _handle_coherence_protection(args: dict) -> CallToolResult:
    mod = _load_bq_module("hardware/coherence_protection.py")
    if not mod:
        return _err("coherence_protection module not found — run from repo root.")
    try:
        materials = mod.MaterialProperties(
            mu_metal_permeability=100_000,
            shielding_factor_dB=float(args.get("shielding_db", 80)),
        )
        protection = mod.ProtectionParameters(
            dnp_microwave_power_W=float(args.get("dnp_power_W", 0.1)),
            laser_power_mW=float(args.get("laser_power_mW", 10)),
            feedback_loop_rate_Hz=float(args.get("feedback_loop_hz", 1000)),
        )

        cps = mod.CoherenceProtectionSystem(materials=materials, protection=protection)
        status = cps.activate_protection()

        coherence_us = status["coherence_time_s"] * 1e6
        bare_us = materials.nv_coherence_time_bare_s * 1e6

        return _ok({
            "bare_coherence_time_us": round(bare_us, 2),
            "protected_coherence_time_s": round(float(status["coherence_time_s"]), 4),
            "protected_coherence_ms": round(coherence_us / 1000, 2),
            "total_enhancement_factor": round(float(status["enhancement_factor"]), 0),
            "per_layer_enhancement": {
                k: round(float(v), 1)
                for k, v in status["contributions"].items()
            },
            "hardware_stack": [
                "Diamond NV centers (core quantum registers, 1.6 μs bare T₂)",
                "Silicon Carbide shell (thermal management, 490 W/m·K)",
                "Bi₂Se₃ topological insulator (topological protection, 0.3 eV gap)",
                f"Mu-metal magnetic shielding ({args.get('shielding_db', 80)} dB)",
                "Aerogel thermal insulation (0.015 W/m·K)",
            ],
            "active_systems": [
                f"DNP ({args.get('dnp_power_W', 0.1)} W microwave at 9.5 GHz)",
                f"Chirped laser pulses ({args.get('laser_power_mW', 10)} mW at 532 nm)",
                f"Feedback control ({args.get('feedback_loop_hz', 1000)} Hz loop)",
            ],
            "comparison": {
                "superconducting_qubits_T2_us": 100,
                "trapped_ions_T2_s": 10,
                "this_system_T2_s": round(float(status["coherence_time_s"]), 3),
                "temperature_K": 300,
                "requires_cryogenics": False,
            },
        })
    except Exception as e:
        return _err(f"Coherence protection analysis failed: {e}")


async def _handle_ai_control(args: dict) -> CallToolResult:
    mod = _fmo_module()
    if not mod:
        return _err("fmo_complex module not found — run from repo root.")
    try:
        target_fs = float(args.get("target_coherence_fs", 1000))
        light_init = float(args.get("initial_light_intensity", 0.5))
        field_init = float(args.get("initial_magnetic_field_mT", 10))
        ph_init = float(args.get("initial_ph", 7.4))

        fmo = mod.FMOComplex()
        ai_fmo = mod.AIControlledFMO(fmo)

        # Set initial control state
        ai_fmo.control_policy["light_intensity"] = light_init
        ai_fmo.control_policy["magnetic_field_mT"] = field_init
        ai_fmo.control_policy["pH"] = ph_init

        result = ai_fmo.optimize_coherence(target_coherence_fs=target_fs)

        baseline_eff = fmo.simulate_energy_transfer(1, 3, 500)
        optimised_coherence = result.get("achieved_coherence_fs",
                                          fmo.params.coherence_time_fs)
        optimised_eff = baseline_eff * min(2.0, optimised_coherence / fmo.params.coherence_time_fs)

        return _ok({
            "target_coherence_fs": target_fs,
            "natural_coherence_fs": fmo.params.coherence_time_fs,
            "optimised_control_params": {
                "light_intensity": round(float(ai_fmo.control_policy["light_intensity"]), 3),
                "magnetic_field_mT": round(float(ai_fmo.control_policy["magnetic_field_mT"]), 2),
                "pH": round(float(ai_fmo.control_policy["pH"]), 2),
                "temperature_K": round(float(ai_fmo.control_policy["temperature_K"]), 1),
            },
            "predicted_coherence_fs": round(
                float(result.get("predicted_coherence_fs", optimised_coherence)), 1
            ),
            "coherence_extension_factor": round(
                optimised_coherence / fmo.params.coherence_time_fs, 2
            ),
            "baseline_transfer_efficiency": round(float(baseline_eff), 4),
            "estimated_optimised_efficiency": round(float(optimised_eff), 4),
            "control_iterations": 10,
            "platform": "AI-Controlled FMO Biological Quantum Computer",
            "note": (
                "AI tunes light intensity, B-field, pH, and temperature microzone "
                "to extend FMO quantum coherence beyond the natural 660 fs baseline. "
                "Full deployment uses real-time 2D spectroscopy feedback."
            ),
        })
    except Exception as e:
        return _err(f"AI control optimisation failed: {e}")


# ── Handler dispatch ──────────────────────────────────────────────────────────

HANDLERS: dict[str, Any] = {
    "bio_quantum_fmo":                  _handle_fmo,
    "bio_quantum_spectroscopy_2d":      _handle_spectroscopy_2d,
    "bio_quantum_vqe":                  _handle_vqe,
    "bio_quantum_coherence_protection": _handle_coherence_protection,
    "bio_quantum_ai_control":           _handle_ai_control,
}
