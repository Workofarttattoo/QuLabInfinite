"""
QuLab MCP Server
================
Exposes QuLab Infinite's scientific labs as Model Context Protocol tools.
Transports supported: stdio (Claude Desktop / claude CLI).

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

# ---------------------------------------------------------------------------
# Lab path bootstrap — works whether installed via pip or run from the repo
# ---------------------------------------------------------------------------

def _find_lab_root() -> Path | None:
    """Return the QuLabInfinite root if we can locate it on sys.path or nearby."""
    # When installed alongside the labs (editable install or monorepo)
    candidates = [
        Path(__file__).parent.parent.parent.parent,  # packages/qulab-mcp/../../.. = repo root
        Path(__file__).parent.parent.parent,
    ]
    for c in candidates:
        if (c / "quantum_computing_lab.py").exists():
            return c
    return None


_LAB_ROOT = _find_lab_root()
if _LAB_ROOT and str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))


def _import_lab(module_name: str):
    """Lazy-import a lab module, returning None on failure."""
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to JSON-serialisable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _ok(data: Any) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(_json_safe(data), indent=2))]
    )


def _err(msg: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": msg}))],
        isError=True,
    )


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    # ── Quantum ──────────────────────────────────────────────────────────────
    Tool(
        name="quantum_bell_state",
        description=(
            "Prepare a Bell (maximally entangled) 2-qubit state and return "
            "the state vector and measurement probabilities. "
            "bell_type: 'phi_plus' | 'phi_minus' | 'psi_plus' | 'psi_minus'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bell_type": {"type": "string", "default": "phi_plus"},
            },
        },
    ),
    Tool(
        name="quantum_grovers_search",
        description=(
            "Run Grover's quantum search algorithm to find a target integer "
            "in an n-qubit search space. Returns the found state and success "
            "probability. n_qubits ≤ 20 recommended (memory grows as 2^n)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "n_qubits": {"type": "integer", "minimum": 2, "maximum": 20},
                "target": {"type": "integer"},
            },
            "required": ["n_qubits", "target"],
        },
    ),
    Tool(
        name="quantum_teleportation",
        description=(
            "Simulate the quantum teleportation protocol for an arbitrary "
            "single-qubit state |ψ⟩ = α|0⟩ + β|1⟩. "
            "Pass alpha_re, alpha_im, beta_re, beta_im (must satisfy |α|²+|β|²=1)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "alpha_re": {"type": "number"},
                "alpha_im": {"type": "number", "default": 0},
                "beta_re": {"type": "number"},
                "beta_im": {"type": "number", "default": 0},
            },
            "required": ["alpha_re", "beta_re"],
        },
    ),
    # ── Particle Physics ─────────────────────────────────────────────────────
    Tool(
        name="particle_cross_section",
        description=(
            "Calculate QED/QCD cross-sections. "
            "process: 'ee_to_mumu' | 'ee_to_hadrons' | 'pp_elastic' | 'deep_inelastic'. "
            "s: centre-of-mass energy squared (GeV²)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "process": {"type": "string"},
                "s": {"type": "number"},
                "cos_theta": {"type": "number"},
            },
            "required": ["process", "s"],
        },
    ),
    Tool(
        name="particle_breit_wigner",
        description=(
            "Breit-Wigner resonance cross-section. "
            "Useful for Z peak (M=91.19 GeV, Gamma=2.495 GeV), "
            "Higgs (M=125.1 GeV, Gamma=0.00407 GeV), etc. "
            "Returns cross-section in GeV⁻² (multiply by 0.3894 for mb)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "E_cm": {"type": "number"},
                "M": {"type": "number"},
                "Gamma": {"type": "number"},
                "J_resonance": {"type": "number", "default": 1},
                "J1": {"type": "number", "default": 0.5},
                "J2": {"type": "number", "default": 0.5},
                "partial_width_in": {"type": "number"},
                "partial_width_out": {"type": "number"},
            },
            "required": ["E_cm", "M", "Gamma", "partial_width_in", "partial_width_out"],
        },
    ),
    Tool(
        name="particle_decay_rate",
        description=(
            "Calculate partial decay rate for a particle. "
            "parent: e.g. 'muon', 'neutron', 'pion+'. "
            "products: list of product names, e.g. ['electron','electron_neutrino','muon_neutrino']."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "parent": {"type": "string"},
                "products": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["parent", "products"],
        },
    ),
    # ── Astrophysics ─────────────────────────────────────────────────────────
    Tool(
        name="astro_lane_emden",
        description=(
            "Solve the Lane-Emden equation for polytropic stellar structure. "
            "n=1.5 → convective star, n=3 → radiative (Eddington) star. "
            "Returns dimensionless radius, θ profile, and the surface value ξ₁."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "n": {"type": "number"},
                "xi_max": {"type": "number", "default": 15.0},
            },
            "required": ["n"],
        },
    ),
    Tool(
        name="astro_cepheid_luminosity",
        description=(
            "Cepheid period-luminosity relation (Leavitt Law). "
            "Returns luminosity in watts and solar luminosities."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "period_days": {"type": "number"},
                "metallicity": {"type": "number", "default": 0.02},
            },
            "required": ["period_days"],
        },
    ),
    Tool(
        name="astro_schwarzschild",
        description=(
            "Compute Schwarzschild metric quantities (gravitational redshift, "
            "time dilation, escape velocity) at radii r outside a mass M. "
            "M in kg, r_values as list of metres."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "M_kg": {"type": "number"},
                "r_values_m": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["M_kg", "r_values_m"],
        },
    ),
    # ── Thermodynamics ───────────────────────────────────────────────────────
    Tool(
        name="thermo_equilibrium_constant",
        description=(
            "Calculate the equilibrium constant K from ΔG° at temperature T. "
            "Uses K = exp(-ΔG° / RT). delta_G in J/mol, T in Kelvin."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "delta_G": {"type": "number"},
                "T": {"type": "number"},
            },
            "required": ["delta_G", "T"],
        },
    ),
    Tool(
        name="thermo_clausius_clapeyron",
        description=(
            "Clausius-Clapeyron equation: vapour pressure at temperature T "
            "given a reference point (T_ref, P_ref) and enthalpy of vaporisation. "
            "Temperatures in Kelvin, pressures in Pa, H_vap in J/mol."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "T": {"type": "number"},
                "T_ref": {"type": "number"},
                "P_ref": {"type": "number"},
                "H_vap": {"type": "number"},
            },
            "required": ["T", "T_ref", "P_ref", "H_vap"],
        },
    ),
    # ── Genomics ─────────────────────────────────────────────────────────────
    Tool(
        name="genomics_align",
        description=(
            "Align two DNA/protein sequences using Needleman-Wunsch (global) "
            "or Smith-Waterman (local) alignment. "
            "mode: 'global' | 'local'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "seq1": {"type": "string"},
                "seq2": {"type": "string"},
                "mode": {"type": "string", "default": "global"},
            },
            "required": ["seq1", "seq2"],
        },
    ),
    Tool(
        name="genomics_call_variants",
        description=(
            "Call SNVs and indels from a set of sequencing reads aligned to a "
            "reference sequence. Returns variants with position, ref/alt alleles, "
            "allele frequency, and genotype."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "reference": {"type": "string"},
                "reads": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["reference", "reads"],
        },
    ),
    # ── Medical / Clinical ───────────────────────────────────────────────────
    Tool(
        name="pharma_pk_model",
        description=(
            "One-compartment pharmacokinetic model. Given a dose (mg) and "
            "PK parameters, returns plasma concentration profile over time. "
            "clearance in L/h, volume_distribution in L, absorption_rate in h⁻¹."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "dose_mg": {"type": "number"},
                "clearance": {"type": "number"},
                "volume_distribution": {"type": "number"},
                "absorption_rate": {"type": "number", "default": 1.0},
                "bioavailability": {"type": "number", "default": 1.0},
                "t_max_h": {"type": "number", "default": 24},
                "iv": {"type": "boolean", "default": True},
            },
            "required": ["dose_mg", "clearance", "volume_distribution"],
        },
    ),
    Tool(
        name="pharma_emax_model",
        description=(
            "Hill / Emax pharmacodynamic model: E = Emax * C^n / (EC50^n + C^n). "
            "Returns effect at the given concentration(s)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "concentration": {
                    "oneOf": [
                        {"type": "number"},
                        {"type": "array", "items": {"type": "number"}},
                    ]
                },
                "emax": {"type": "number"},
                "ec50": {"type": "number"},
                "hill_coefficient": {"type": "number", "default": 1.0},
                "baseline": {"type": "number", "default": 0.0},
            },
            "required": ["concentration", "emax", "ec50"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Handler implementations
# ---------------------------------------------------------------------------

async def _handle_quantum_bell_state(args: dict) -> CallToolResult:
    mod = _import_lab("quantum_computing_lab")
    if not mod:
        return _err("quantum_computing_lab not found — install qulab-infinite or run from the repo root.")
    lab = mod.QuantumComputingLab()
    bell_type = args.get("bell_type", "phi_plus")
    state = lab.bell_state_preparation(bell_type)
    probs = np.abs(state) ** 2
    return _ok({
        "bell_type": bell_type,
        "state_vector": {"real": state.real.tolist(), "imag": state.imag.tolist()},
        "probabilities": probs.tolist(),
        "basis_labels": ["|00⟩", "|01⟩", "|10⟩", "|11⟩"],
        "entanglement_entropy_bits": float(lab.calculate_entanglement_entropy(state, [0])),
    })


async def _handle_quantum_grovers(args: dict) -> CallToolResult:
    mod = _import_lab("quantum_computing_lab")
    if not mod:
        return _err("quantum_computing_lab not found.")
    n = int(args["n_qubits"])
    target = int(args["target"])
    if target >= 2 ** n:
        return _err(f"target {target} out of range for {n}-qubit space (max {2**n - 1}).")
    lab = mod.QuantumComputingLab()
    oracle = lambda x: x == target
    found, prob = lab.grovers_algorithm(oracle, n)
    return _ok({
        "n_qubits": n,
        "target": target,
        "found_state": found,
        "found_decimal": int(found, 2),
        "success_probability": prob,
        "optimal_iterations": int(np.round(np.pi / 4 * np.sqrt(2 ** n))),
    })


async def _handle_quantum_teleportation(args: dict) -> CallToolResult:
    mod = _import_lab("quantum_computing_lab")
    if not mod:
        return _err("quantum_computing_lab not found.")
    alpha = complex(args["alpha_re"], args.get("alpha_im", 0))
    beta = complex(args["beta_re"], args.get("beta_im", 0))
    norm = abs(alpha) ** 2 + abs(beta) ** 2
    if abs(norm - 1.0) > 0.01:
        return _err(f"|α|²+|β|²={norm:.4f} must equal 1 (state must be normalised).")
    psi = np.array([alpha, beta], dtype=complex) / np.sqrt(norm)
    lab = mod.QuantumComputingLab()
    measurement, bob_state = lab.quantum_teleportation(psi)
    fidelity = lab.calculate_fidelity(psi, bob_state)
    return _ok({
        "input_state": {"alpha": [alpha.real, alpha.imag], "beta": [beta.real, beta.imag]},
        "alice_measurement": measurement,
        "bob_state": {"real": bob_state.real.tolist(), "imag": bob_state.imag.tolist()},
        "fidelity": fidelity,
        "teleportation_successful": bool(fidelity > 0.999),
    })


async def _handle_particle_cross_section(args: dict) -> CallToolResult:
    mod = _import_lab("particle_physics_lab")
    if not mod:
        return _err("particle_physics_lab not found.")
    lab = mod.ParticlePhysicsLab()
    sigma = lab.calculate_cross_section(
        args["process"], args["s"],
        cos_theta=args.get("cos_theta"),
    )
    return _ok({"process": args["process"], "s_GeV2": args["s"], "sigma_mb": sigma})


async def _handle_particle_breit_wigner(args: dict) -> CallToolResult:
    mod = _import_lab("particle_physics_lab")
    if not mod:
        return _err("particle_physics_lab not found.")
    lab = mod.ParticlePhysicsLab()
    sigma = lab.breit_wigner_cross_section(
        E_cm=args["E_cm"], M=args["M"], Gamma=args["Gamma"],
        J_resonance=args.get("J_resonance", 1),
        J1=args.get("J1", 0.5), J2=args.get("J2", 0.5),
        partial_width_in=args["partial_width_in"],
        partial_width_out=args["partial_width_out"],
    )
    return _ok({
        "E_cm_GeV": args["E_cm"],
        "M_GeV": args["M"],
        "sigma_GeV-2": sigma,
        "sigma_mb": sigma * 0.3894,
        "sigma_nb": sigma * 3.894e5,
        "sigma_pb": sigma * 3.894e8,
    })


async def _handle_particle_decay(args: dict) -> CallToolResult:
    mod = _import_lab("particle_physics_lab")
    if not mod:
        return _err("particle_physics_lab not found.")
    lab = mod.ParticlePhysicsLab()
    try:
        rate = lab.calculate_decay_rate(args["parent"], args["products"])
        br = lab.calculate_branching_ratio(args["parent"], args["products"])
        return _ok({"parent": args["parent"], "products": args["products"],
                    "decay_rate_GeV": rate, "branching_ratio": br})
    except (KeyError, ValueError) as e:
        return _err(str(e))


async def _handle_astro_lane_emden(args: dict) -> CallToolResult:
    mod = _import_lab("astrophysics_lab")
    if not mod:
        return _err("astrophysics_lab not found.")
    lab = mod.AstrophysicsLab()
    r = lab.lane_emden_solver(n=args["n"], xi_max=args.get("xi_max", 15.0))
    return _ok({
        "n": args["n"],
        "xi_1": float(r["xi_1"]),
        "dtheta_1": float(r["dtheta_1"]),
        "xi": r["xi"].tolist(),
        "theta": r["theta"].tolist(),
    })


async def _handle_astro_cepheid(args: dict) -> CallToolResult:
    mod = _import_lab("astrophysics_lab")
    if not mod:
        return _err("astrophysics_lab not found.")
    lab = mod.AstrophysicsLab()
    L_W = lab.cepheid_period_luminosity(args["period_days"], args.get("metallicity", 0.02))
    return _ok({
        "period_days": args["period_days"],
        "luminosity_watts": L_W,
        "luminosity_solar": L_W / lab.L_sun,
    })


async def _handle_astro_schwarzschild(args: dict) -> CallToolResult:
    mod = _import_lab("astrophysics_lab")
    if not mod:
        return _err("astrophysics_lab not found.")
    lab = mod.AstrophysicsLab()
    r = np.array(args["r_values_m"])
    result = lab.schwarzschild_metric(M=args["M_kg"], r=r)
    return _ok(result)


async def _handle_thermo_equilibrium(args: dict) -> CallToolResult:
    mod = _import_lab("thermodynamics_lab")
    if not mod:
        return _err("thermodynamics_lab not found.")
    lab = mod.ThermodynamicsLab()
    K = lab.equilibrium_constant(args["delta_G"], args["T"])
    return _ok({"delta_G_J_per_mol": args["delta_G"], "T_K": args["T"], "K": K, "ln_K": float(np.log(K))})


async def _handle_thermo_clausius_clapeyron(args: dict) -> CallToolResult:
    mod = _import_lab("thermodynamics_lab")
    if not mod:
        return _err("thermodynamics_lab not found.")
    lab = mod.ThermodynamicsLab()
    P = lab.vapor_pressure_clausius_clapeyron(args["T"], args["T_ref"], args["P_ref"], args["H_vap"])
    return _ok({"T_K": args["T"], "vapor_pressure_Pa": P, "vapor_pressure_kPa": P / 1000,
                "vapor_pressure_bar": P / 1e5, "vapor_pressure_atm": P / 101325})


async def _handle_genomics_align(args: dict) -> CallToolResult:
    mod = _import_lab("genomics_lab")
    if not mod:
        return _err("genomics_lab not found.")
    lab = mod.GenomicsLab()
    result = lab.align_sequences(args["seq1"], args["seq2"], mode=args.get("mode", "global"))
    return _ok(result)


async def _handle_genomics_variants(args: dict) -> CallToolResult:
    mod = _import_lab("genomics_lab")
    if not mod:
        return _err("genomics_lab not found.")
    lab = mod.GenomicsLab()
    result = lab.call_variants(args["reference"], args["reads"])
    # Convert Variant dataclasses to dicts
    result["snvs"] = [
        {"position": v.position, "ref": v.ref, "alt": v.alt,
         "allele_frequency": v.allele_frequency, "genotype": v.genotype,
         "quality": v.quality, "type": v.variant_type}
        for v in result["snvs"]
    ]
    result["indels"] = [
        {"position": v.position, "ref": v.ref, "alt": v.alt,
         "type": v.variant_type}
        for v in result["indels"]
    ]
    return _ok(result)


async def _handle_pharma_pk(args: dict) -> CallToolResult:
    mod = _import_lab("pharmacology_lab")
    if not mod:
        return _err("pharmacology_lab not found.")
    lab = mod.PharmacologyLab()
    pk = mod.PKParameters(
        clearance=args["clearance"],
        volume_distribution=args["volume_distribution"],
        absorption_rate=args.get("absorption_rate", 1.0),
        bioavailability=args.get("bioavailability", 1.0),
    )
    t_max = args.get("t_max_h", 24)
    t = np.linspace(0, t_max, 200)
    conc = lab.one_compartment_model(
        dose=args["dose_mg"], params=pk, time=t, iv=args.get("iv", True)
    )
    cmax = float(np.max(conc))
    tmax_idx = int(np.argmax(conc))
    return _ok({
        "dose_mg": args["dose_mg"],
        "half_life_h": pk.half_life,
        "Cmax_mg_L": cmax,
        "Tmax_h": float(t[tmax_idx]),
        "AUC_mg_h_L": float(np.trapezoid(conc, t) if hasattr(np, "trapezoid") else np.trapz(conc, t)),
        "time_h": t.tolist(),
        "concentration_mg_L": conc.tolist(),
    })


async def _handle_pharma_emax(args: dict) -> CallToolResult:
    mod = _import_lab("pharmacology_lab")
    if not mod:
        return _err("pharmacology_lab not found.")
    lab = mod.PharmacologyLab()
    pd = mod.PDParameters(
        emax=args["emax"],
        ec50=args["ec50"],
        hill_coefficient=args.get("hill_coefficient", 1.0),
        baseline=args.get("baseline", 0.0),
    )
    c = args["concentration"]
    conc_arr = np.array(c if isinstance(c, list) else [c])
    effects = lab.emax_model(concentration=conc_arr, params=pd)
    return _ok({
        "concentrations": conc_arr.tolist(),
        "effects": effects.tolist() if hasattr(effects, "tolist") else [float(effects)],
        "emax": args["emax"], "ec50": args["ec50"],
        "hill_coefficient": args.get("hill_coefficient", 1.0),
    })


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_HANDLERS = {
    "quantum_bell_state": _handle_quantum_bell_state,
    "quantum_grovers_search": _handle_quantum_grovers,
    "quantum_teleportation": _handle_quantum_teleportation,
    "particle_cross_section": _handle_particle_cross_section,
    "particle_breit_wigner": _handle_particle_breit_wigner,
    "particle_decay_rate": _handle_particle_decay,
    "astro_lane_emden": _handle_astro_lane_emden,
    "astro_cepheid_luminosity": _handle_astro_cepheid,
    "astro_schwarzschild": _handle_astro_schwarzschild,
    "thermo_equilibrium_constant": _handle_thermo_equilibrium,
    "thermo_clausius_clapeyron": _handle_thermo_clausius_clapeyron,
    "genomics_align": _handle_genomics_align,
    "genomics_call_variants": _handle_genomics_variants,
    "pharma_pk_model": _handle_pharma_pk,
    "pharma_emax_model": _handle_pharma_emax,
}


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

def create_server() -> Server:
    server = Server("qulab-infinite")

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> CallToolResult:
        handler = _HANDLERS.get(name)
        if handler is None:
            return _err(f"Unknown tool: {name!r}. Use list_tools to see available tools.")
        try:
            return await handler(arguments or {})
        except Exception as exc:
            return _err(f"{type(exc).__name__}: {exc}")

    return server


async def run() -> None:
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
