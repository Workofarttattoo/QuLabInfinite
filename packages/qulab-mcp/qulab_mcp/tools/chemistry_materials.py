"""
Chemistry & Materials Science MCP tools for QuLab Infinite.

Covers:
  - Computational chemistry  (molecular energy, geometry optimisation)
  - Inorganic chemistry      (lattice energy, band gap, crystal field, redox)
  - Electrochemistry         (Nernst, Tafel, Butler-Volmer)
  - Physical chemistry       (kinetic theory, Carnot)
  - Catalysis                (Langmuir-Hinshelwood dynamics)
  - Polymer science          (random-walk, dielectric screening)
  - Materials database       (1 619 materials, 18 categories)
  - Nanotechnology           (Brus, Ostwald ripening, surface area)
  - Semiconductor devices    (MOSFET, p-n junction, quantum well, diffusion)

Copyright © 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
All Rights Reserved. PATENT PENDING.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from mcp.types import CallToolResult, TextContent, Tool

# ── helpers ──────────────────────────────────────────────────────────────────

def _ok(data) -> CallToolResult:
    import json
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(_safe(data), indent=2))])

def _err(msg: str) -> CallToolResult:
    import json
    return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": msg}))], isError=True)

def _safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    return obj

def _import(dotted: str):
    try:
        return importlib.import_module(dotted)
    except Exception:
        return None

def _lab_root() -> Path | None:
    candidates = [
        Path(__file__).parents[4],          # packages/qulab-mcp/../../../.. = repo root
        Path(__file__).parents[3],
    ]
    for c in candidates:
        if (c / "computational_chemistry_lab.py").exists():
            return c
    return None

_LAB_ROOT = _lab_root()
if _LAB_ROOT and str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

# Lazy-loaded materials database (heavy; loads once)
_mat_db_instance = None

def _get_materials_db():
    global _mat_db_instance
    if _mat_db_instance is not None:
        return _mat_db_instance
    if _LAB_ROOT is None:
        return None
    db_path = _LAB_ROOT / "qulab/labs/engineering/materials_lab/materials_database.py"
    if not db_path.exists():
        return None
    # The package __init__ has a broken import; load the file directly.
    if "qulab.labs.engineering.materials_lab" not in sys.modules:
        stub = types.ModuleType("qulab.labs.engineering.materials_lab")
        sys.modules["qulab.labs.engineering.materials_lab"] = stub
    mat_path = _LAB_ROOT / "qulab/labs/engineering/materials_lab"
    if str(mat_path) not in sys.path:
        sys.path.insert(0, str(mat_path))
    spec = importlib.util.spec_from_file_location("_mat_db_direct", str(db_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _mat_db_instance = mod.MaterialsDatabase()
    return _mat_db_instance


# ── Tool definitions ─────────────────────────────────────────────────────────

TOOLS: list[Tool] = [

    # ── Computational Chemistry ──────────────────────────────────────────────
    Tool(
        name="chem_molecular_energy",
        description=(
            "Calculate molecular energy using Molecular Mechanics (MM), "
            "semi-empirical AM1, or a simplified DFT functional. "
            "Provide atoms as a list of {symbol, x, y, z} dicts (Å). "
            "method: 'MM' | 'AM1' | 'DFT'. functional: 'LDA' | 'PBE' (DFT only)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "atoms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string"},
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                        },
                        "required": ["symbol", "x", "y", "z"],
                    },
                },
                "method": {"type": "string", "default": "MM"},
                "functional": {"type": "string", "default": "LDA"},
                "charge": {"type": "integer", "default": 0},
                "temperature": {"type": "number", "default": 298.15},
            },
            "required": ["atoms"],
        },
    ),

    # ── Inorganic Chemistry ───────────────────────────────────────────────────
    Tool(
        name="chem_lattice_energy",
        description=(
            "Born-Haber lattice energy calculation. "
            "q1, q2: ion charges (e.g. +2, -2). "
            "r0: equilibrium interionic distance (Å). "
            "epsilon: Madelung constant × geometric factor (default 9.0 for NaCl type). "
            "n: Born exponent (default 8.5)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "q1": {"type": "integer"},
                "q2": {"type": "integer"},
                "r0": {"type": "number"},
                "epsilon": {"type": "number", "default": 9.0},
                "n": {"type": "number", "default": 8.5},
            },
            "required": ["q1", "q2", "r0"],
        },
    ),
    Tool(
        name="chem_band_gap",
        description=(
            "Calculate semiconductor band gap from valence-band maximum (e1) "
            "and conduction-band minimum (e2) energies in eV. "
            "k_value is the Boltzmann constant in eV/K (default 8.617e-5)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "e1": {"type": "number"},
                "e2": {"type": "number"},
                "k_value": {"type": "number", "default": 8.617333262e-5},
            },
            "required": ["e1", "e2"],
        },
    ),
    Tool(
        name="chem_crystal_field",
        description=(
            "Crystal field splitting energy (Δ) for a transition metal complex. "
            "d_orbital_population: number of d electrons (1–10). "
            "octahedral_field_strength and tetrahedral_field_strength in eV."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "d_orbital_population": {"type": "integer", "minimum": 1, "maximum": 10},
                "octahedral_field_strength": {"type": "number", "default": 0.429},
                "tetrahedral_field_strength": {"type": "number", "default": 1.735},
            },
            "required": ["d_orbital_population"],
        },
    ),
    Tool(
        name="chem_redox_potential",
        description=(
            "Calculate electrochemical cell potential (E_cell) from standard "
            "electrode potentials. Provide a dict of half-reaction potentials "
            "(e.g. {'Zn2+/Zn': -0.76, 'Cu2+/Cu': +0.34}) plus the anode and "
            "cathode half-reaction keys."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "electrode_potentials": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "anode": {"type": "string"},
                "cathode": {"type": "string"},
            },
            "required": ["electrode_potentials", "anode", "cathode"],
        },
    ),
    Tool(
        name="chem_activation_energy",
        description=(
            "Arrhenius equation: given activation energy Ea (J/mol) and two "
            "temperatures T1, T2 (K), return the ratio k2/k1 and the individual "
            "rate scaling factors."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "Ea": {"type": "number"},
                "T1": {"type": "number"},
                "T2": {"type": "number"},
            },
            "required": ["Ea", "T1", "T2"],
        },
    ),

    # ── Electrochemistry ──────────────────────────────────────────────────────
    Tool(
        name="chem_nernst_potential",
        description=(
            "Nernst equation: E = E° - (RT/nF) ln(Q). "
            "e0: standard potential (V). n: electrons transferred. "
            "concentration_ox / concentration_red: concentrations (mol/L). "
            "temperature: K (default 298.15)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "e0": {"type": "number"},
                "n": {"type": "integer"},
                "concentration_ox": {"type": "number"},
                "concentration_red": {"type": "number"},
                "temperature": {"type": "number", "default": 298.15},
            },
            "required": ["e0", "n", "concentration_ox", "concentration_red"],
        },
    ),

    # ── Physical Chemistry ────────────────────────────────────────────────────
    Tool(
        name="chem_kinetic_rms_velocity",
        description=(
            "Kinetic theory of gases: root-mean-square velocity = sqrt(3RT/M). "
            "temperature in K, molar_mass in kg/mol."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "molar_mass": {"type": "number"},
            },
            "required": ["temperature", "molar_mass"],
        },
    ),
    Tool(
        name="chem_carnot_efficiency",
        description=(
            "Carnot cycle maximum thermodynamic efficiency = 1 - T_cold/T_hot. "
            "Both temperatures in Kelvin."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "t_hot": {"type": "number"},
                "t_cold": {"type": "number"},
            },
            "required": ["t_hot", "t_cold"],
        },
    ),

    # ── Catalysis ─────────────────────────────────────────────────────────────
    Tool(
        name="chem_catalysis_simulate",
        description=(
            "Simulate Langmuir-Hinshelwood catalytic dynamics. "
            "temperature: K. pressure: Pa. "
            "initial_concentration: list of [reactant, product] mol/L. "
            "rate_constants: dict of 'forward' and 'reverse' (s⁻¹). "
            "time_steps / dt: simulation resolution."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "default": 298.15},
                "pressure": {"type": "number", "default": 101325},
                "initial_concentration": {
                    "type": "array",
                    "items": {"type": "number"},
                    "default": [1.0, 0.0],
                },
                "rate_constants": {
                    "type": "object",
                    "properties": {
                        "forward": {"type": "number"},
                        "reverse": {"type": "number"},
                    },
                    "default": {"forward": 0.1, "reverse": 0.01},
                },
                "time_steps": {"type": "integer", "default": 200},
                "dt": {"type": "number", "default": 0.01},
            },
        },
    ),

    # ── Polymer Science ───────────────────────────────────────────────────────
    Tool(
        name="chem_polymer_properties",
        description=(
            "Polymer chain statistics. "
            "monomer_mass: g/mol. n_monomers: chain length. "
            "temperature: K. solvent_permittivity: relative permittivity (78 for water). "
            "Returns end-to-end distance (nm), radius of gyration, dielectric screening."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "monomer_mass": {"type": "number", "default": 150.0},
                "n_monomers": {"type": "integer", "default": 100},
                "temperature": {"type": "number", "default": 300.0},
                "solvent_permittivity": {"type": "number", "default": 78.0},
            },
        },
    ),

    # ── Materials Database ────────────────────────────────────────────────────
    Tool(
        name="materials_lookup",
        description=(
            "Look up a material from the 1 619-entry QuLab materials database. "
            "Returns density, mechanical properties (Young's modulus, yield strength, "
            "tensile strength), thermal properties, and more. "
            "Examples: 'steel', 'aluminum', 'silicon', 'graphene', 'PTFE'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="materials_search",
        description=(
            "Search the 1 619-material database by name text and/or property ranges. "
            "category: one of '2D_material', 'biomaterial', 'ceramic', 'composite', "
            "'element', 'energy_material', 'magnetic_material', 'metal', 'nanomaterial', "
            "'optical_material', 'polymer', 'semiconductor', 'superconductor', "
            "'thermal_material'. text: free-text name search. All numeric filters optional."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "category": {"type": "string"},
                "subcategory": {"type": "string"},
                "min_density": {"type": "number"},
                "max_density": {"type": "number"},
                "min_strength": {"type": "number"},
                "max_strength": {"type": "number"},
                "min_youngs_modulus": {"type": "number"},
                "max_cost": {"type": "number"},
                "availability": {"type": "string"},
                "max_results": {"type": "integer", "default": 10},
            },
        },
    ),
    Tool(
        name="materials_categories",
        description="List all categories and subcategories available in the materials database.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="materials_design",
        description=(
            "Compute derived mechanical / thermal design properties for a material "
            "from its elastic constants matrix (3×3 or 6×6 Voigt notation). "
            "elastic_constants: flat list of 9 values (row-major 3×3). "
            "Returns bulk modulus, shear modulus, Poisson's ratio (Voigt averages)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "elastic_constants": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 9,
                    "maxItems": 9,
                    "description": "9 values (C11,C12,C13,C21,C22,C23,C31,C32,C33) in GPa",
                },
            },
            "required": ["elastic_constants"],
        },
    ),

    # ── Nanotechnology ────────────────────────────────────────────────────────
    Tool(
        name="nano_quantum_dot_bandgap",
        description=(
            "Brus equation: quantum confinement bandgap for a spherical quantum dot. "
            "radius_nm: dot radius. bulk_bandgap_eV: bulk semiconductor bandgap. "
            "electron_mass_ratio / hole_mass_ratio: effective mass ratios (m*/m_e). "
            "dielectric_constant: relative permittivity of the material."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "radius_nm": {"type": "number"},
                "bulk_bandgap_eV": {"type": "number"},
                "electron_mass_ratio": {"type": "number", "default": 0.067},
                "hole_mass_ratio": {"type": "number", "default": 0.45},
                "dielectric_constant": {"type": "number", "default": 12.9},
            },
            "required": ["radius_nm", "bulk_bandgap_eV"],
        },
    ),
    Tool(
        name="nano_surface_area",
        description=(
            "Calculate specific surface area (BET-style) for spherical nanoparticles. "
            "diameter_nm: particle diameter. density_g_per_cm3: material density. "
            "porosity: fraction of void space (0–1, default 0)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "diameter_nm": {"type": "number"},
                "density_g_per_cm3": {"type": "number"},
                "porosity": {"type": "number", "default": 0.0},
            },
            "required": ["diameter_nm", "density_g_per_cm3"],
        },
    ),
    Tool(
        name="nano_melting_point_depression",
        description=(
            "Gibbs-Thomson equation: melting point depression for nanoparticles. "
            "bulk_melting_K, diameter_nm, surface_energy_J_per_m2, "
            "density_g_per_cm3, heat_of_fusion_kJ_per_mol, molar_mass_g_per_mol."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bulk_melting_K": {"type": "number"},
                "diameter_nm": {"type": "number"},
                "surface_energy_J_per_m2": {"type": "number"},
                "density_g_per_cm3": {"type": "number"},
                "heat_of_fusion_kJ_per_mol": {"type": "number"},
                "molar_mass_g_per_mol": {"type": "number"},
            },
            "required": ["bulk_melting_K", "diameter_nm", "surface_energy_J_per_m2",
                         "density_g_per_cm3", "heat_of_fusion_kJ_per_mol", "molar_mass_g_per_mol"],
        },
    ),
    Tool(
        name="nano_ostwald_ripening",
        description=(
            "Simulate Ostwald ripening of nanoparticles over time. "
            "initial_diameters_nm: list of starting particle diameters. "
            "temperature_K, time_hours. "
            "Returns final size distribution and mean diameter evolution."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "initial_diameters_nm": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "temperature_K": {"type": "number"},
                "time_hours": {"type": "number"},
                "surface_tension": {"type": "number", "default": 1.5},
                "diffusion_coefficient": {"type": "number", "default": 1e-12},
            },
            "required": ["initial_diameters_nm", "temperature_K", "time_hours"],
        },
    ),
    Tool(
        name="nano_drug_release",
        description=(
            "Korsmeyer-Peppas drug release kinetics from nanoparticles. "
            "drug_loading_mg, particle_diameter_nm, release_exponent (n), "
            "rate_constant (k), t_max_hours."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "drug_loading_mg": {"type": "number"},
                "particle_diameter_nm": {"type": "number"},
                "release_exponent": {"type": "number", "default": 0.5},
                "rate_constant": {"type": "number", "default": 0.1},
                "t_max_hours": {"type": "number", "default": 24},
            },
            "required": ["drug_loading_mg", "particle_diameter_nm"],
        },
    ),

    # ── Semiconductor Devices ─────────────────────────────────────────────────
    Tool(
        name="semi_mosfet_iv",
        description=(
            "MOSFET I-V characteristics (drain current vs Vgs and Vds). "
            "V_th: threshold voltage (V). mu_n: electron mobility (cm²/V·s). "
            "C_ox: oxide capacitance per area (F/cm²). W_L_ratio: W/L. "
            "V_gs_range / V_ds_range: [min, max] in volts."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "V_th": {"type": "number"},
                "mu_n": {"type": "number", "default": 450},
                "C_ox": {"type": "number", "default": 3.45e-7},
                "W_L_ratio": {"type": "number", "default": 10},
                "V_gs_range": {
                    "type": "array", "items": {"type": "number"},
                    "default": [0, 3], "minItems": 2, "maxItems": 2,
                },
                "V_ds_range": {
                    "type": "array", "items": {"type": "number"},
                    "default": [0, 3], "minItems": 2, "maxItems": 2,
                },
                "n_points": {"type": "integer", "default": 20},
            },
            "required": ["V_th"],
        },
    ),
    Tool(
        name="semi_threshold_voltage",
        description=(
            "Calculate MOSFET threshold voltage from process parameters. "
            "oxide_thickness_nm, substrate_doping_cm3 (N_a for p-substrate), "
            "metal_work_function_eV (default 4.5 for polysilicon), temperature_K."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "oxide_thickness_nm": {"type": "number"},
                "substrate_doping_cm3": {"type": "number"},
                "oxide_charge_cm2": {"type": "number", "default": 1e10},
                "metal_work_function_eV": {"type": "number", "default": 4.5},
                "temperature_K": {"type": "number", "default": 300},
            },
            "required": ["oxide_thickness_nm", "substrate_doping_cm3"],
        },
    ),
    Tool(
        name="semi_pn_junction",
        description=(
            "p-n junction: built-in potential, depletion width, and capacitance. "
            "N_a_cm3: acceptor doping, N_d_cm3: donor doping, temperature_K."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "N_a_cm3": {"type": "number"},
                "N_d_cm3": {"type": "number"},
                "temperature_K": {"type": "number", "default": 300},
            },
            "required": ["N_a_cm3", "N_d_cm3"],
        },
    ),
    Tool(
        name="semi_quantum_well",
        description=(
            "Quantum well energy levels (particle-in-a-box with finite barriers). "
            "well_width_nm, barrier_height_eV, effective_mass_ratio (m*/m_e, "
            "default 0.067 for GaAs)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "well_width_nm": {"type": "number"},
                "barrier_height_eV": {"type": "number"},
                "effective_mass_ratio": {"type": "number", "default": 0.067},
            },
            "required": ["well_width_nm", "barrier_height_eV"],
        },
    ),
    Tool(
        name="semi_diffusion_profile",
        description=(
            "Dopant diffusion profile (erfc solution to Fick's 2nd law). "
            "depth_range_um: [min, max] µm. surface_concentration_cm3, "
            "diffusion_time_hours, diffusion_coefficient_cm2_per_s, temperature_K."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "depth_range_um": {
                    "type": "array", "items": {"type": "number"},
                    "default": [0, 1], "minItems": 2, "maxItems": 2,
                },
                "surface_concentration_cm3": {"type": "number"},
                "diffusion_time_hours": {"type": "number"},
                "diffusion_coefficient_cm2_per_s": {"type": "number", "default": 1e-13},
                "temperature_K": {"type": "number", "default": 1273},
                "n_points": {"type": "integer", "default": 50},
            },
            "required": ["surface_concentration_cm3", "diffusion_time_hours"],
        },
    ),
]


# ── Handlers ──────────────────────────────────────────────────────────────────

async def _handle_chem_molecular_energy(args: dict) -> CallToolResult:
    mod = _import("computational_chemistry_lab")
    if not mod:
        return _err("computational_chemistry_lab not found — run from the repo root.")
    try:
        lab = mod.ComputationalChemistryLab(temperature=args.get("temperature", 298.15))
        atoms = []
        for a in args["atoms"]:
            sym = a["symbol"]
            an = lab.atomic_numbers.get(sym, 6)
            am = lab.atomic_masses.get(sym, 12.0)
            pos = np.array([a.get("x", 0.0), a.get("y", 0.0), a.get("z", 0.0)])
            atoms.append(mod.Atom(element=sym, position=pos, atomic_number=an, mass=am))
        mol = mod.Molecule(atoms=atoms, charge=args.get("charge", 0))
        method = args.get("method", "MM").upper()
        if method == "AM1":
            energy = lab.am1_energy(mol)
        elif method == "DFT":
            energy = lab.simple_dft_energy(mol, functional=args.get("functional", "LDA"))
        else:
            energy = lab.molecular_mechanics_energy(mol)
        return _ok({"method": method, "energy_kJ_mol": _safe(energy),
                    "n_atoms": len(atoms), "charge": mol.charge})
    except Exception as e:
        return _err(str(e))


async def _handle_chem_lattice_energy(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.chemistry.inorganic")
    if not mod:
        return _err("qulab.labs.chemistry.inorganic not found.")
    lab = mod.InorganicChemistryLab()
    result = lab.lattice_energy_calculator(
        q1=args["q1"], q2=args["q2"], r0=args["r0"],
        epsilon=args.get("epsilon", 9.0), n=args.get("n", 8.5),
    )
    return _ok({"q1": args["q1"], "q2": args["q2"], "r0_angstrom": args["r0"],
                "lattice_energy": _safe(result)})


async def _handle_chem_band_gap(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.chemistry.inorganic")
    if not mod:
        return _err("qulab.labs.chemistry.inorganic not found.")
    lab = mod.InorganicChemistryLab()
    result = lab.band_gap_calculator(
        e1=args["e1"], e2=args["e2"],
        k_value=args.get("k_value", 8.617333262e-5),
    )
    return _ok({"valence_band_max_eV": args["e1"], "conduction_band_min_eV": args["e2"],
                "band_gap_result": _safe(result)})


async def _handle_chem_crystal_field(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.chemistry.inorganic")
    if not mod:
        return _err("qulab.labs.chemistry.inorganic not found.")
    lab = mod.InorganicChemistryLab()
    result = lab.crystal_field_splitting_energy(
        d_orbital_population=args["d_orbital_population"],
        octahedral_field_strength=args.get("octahedral_field_strength", 0.429),
        tetrahedral_field_strength=args.get("tetrahedral_field_strength", 1.735),
    )
    return _ok({"d_electrons": args["d_orbital_population"],
                "crystal_field_splitting_eV": _safe(result)})


async def _handle_chem_redox_potential(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.chemistry.inorganic")
    if not mod:
        return _err("qulab.labs.chemistry.inorganic not found.")
    lab = mod.InorganicChemistryLab()
    result = lab.redox_potential(
        standard_electrode_potentials=args["electrode_potentials"],
        anode=args["anode"],
        cathode=args["cathode"],
    )
    e_anode = args["electrode_potentials"][args["anode"]]
    e_cathode = args["electrode_potentials"][args["cathode"]]
    return _ok({
        "anode": args["anode"], "E_anode_V": e_anode,
        "cathode": args["cathode"], "E_cathode_V": e_cathode,
        "E_cell_V": _safe(result),
        "spontaneous": float(_safe(result)) > 0 if not isinstance(result, np.ndarray) else None,
    })


async def _handle_chem_activation_energy(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.chemistry.inorganic")
    if not mod:
        return _err("qulab.labs.chemistry.inorganic not found.")
    lab = mod.InorganicChemistryLab()
    result = lab.activation_energy(
        Ea=args["Ea"], T1=args["T1"], T2=args["T2"],
    )
    R = 8.314
    ratio = float(np.exp(-args["Ea"] / R * (1 / args["T2"] - 1 / args["T1"])))
    return _ok({"Ea_J_per_mol": args["Ea"], "T1_K": args["T1"], "T2_K": args["T2"],
                "rate_ratio_k2_over_k1": ratio, "raw_result": _safe(result)})


async def _handle_chem_nernst(args: dict) -> CallToolResult:
    R, F = 8.314, 96485.0
    T = args.get("temperature", 298.15)
    n = args["n"]
    e0 = args["e0"]
    Q = args["concentration_ox"] / max(args["concentration_red"], 1e-30)
    E = e0 - (R * T / (n * F)) * np.log(Q)
    return _ok({"E0_V": e0, "n": n, "Q": Q, "T_K": T, "E_V": float(E),
                "formula": "E = E0 - (RT/nF)*ln(Q)"})


async def _handle_chem_kinetic_rms_velocity(args: dict) -> CallToolResult:
    # v_rms = sqrt(3RT/M)  where R=8.314 J/(mol·K), M in kg/mol
    R = 8.314462618  # J/(mol·K)
    T = float(args["temperature"])
    M = float(args["molar_mass"])  # kg/mol
    if M <= 0:
        return _err("molar_mass must be positive (kg/mol).")
    v_rms = float(np.sqrt(3 * R * T / M))
    mean_v = float(np.sqrt(8 * R * T / (np.pi * M)))
    most_probable_v = float(np.sqrt(2 * R * T / M))
    return _ok({
        "temperature_K": T,
        "molar_mass_kg_per_mol": M,
        "v_rms_m_per_s": v_rms,
        "v_mean_m_per_s": mean_v,
        "v_most_probable_m_per_s": most_probable_v,
        "formula": "v_rms = sqrt(3RT/M)",
    })


async def _handle_chem_carnot(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.chemistry.physical")
    if not mod:
        return _err("qulab.labs.chemistry.physical not found.")
    from qulab.labs.chemistry.physical import PhysicalConstants, Thermodynamics
    pc = PhysicalConstants()
    thermo = Thermodynamics(pc)
    if args["t_hot"] <= args["t_cold"]:
        return _err("t_hot must be greater than t_cold for a heat engine.")
    eta = thermo.ideal_carnot_efficiency(t_hot=args["t_hot"], t_cold=args["t_cold"])
    return _ok({"t_hot_K": args["t_hot"], "t_cold_K": args["t_cold"],
                "carnot_efficiency": _safe(eta),
                "max_COP_heat_pump": float(_safe(1 / (1 - args["t_cold"] / args["t_hot"])))})


async def _handle_chem_catalysis(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.chemistry.catalysis")
    if not mod:
        return _err("qulab.labs.chemistry.catalysis not found.")
    conc0 = args.get("initial_concentration", [1.0, 0.0])
    # Map forward/reverse → k1/k2 (underlying lab uses k1/k2 keys)
    rc_in = args.get("rate_constants", {"forward": 0.1, "reverse": 0.01})
    rc = {
        "k1": rc_in.get("k1", rc_in.get("forward", 0.1)),
        "k2": rc_in.get("k2", rc_in.get("reverse", 0.01)),
    }
    lab = mod.CatalysisLab(
        temperature=args.get("temperature", 298.15),
        pressure=args.get("pressure", 101325),
        concentration=np.array(conc0),
        rate_constants=rc,
    )
    t_steps = min(args.get("time_steps", 200), 1000)
    dt = args.get("dt", 0.01)
    time_arr, conc_arr = lab.simulate_catalysis_process(time_steps=t_steps, dt=dt)
    return _ok({
        "temperature_K": args.get("temperature", 298.15),
        "k1_forward": rc["k1"],
        "k2_reverse": rc["k2"],
        "time_s": time_arr.tolist(),
        "concentration": conc_arr.tolist(),
        "final_concentration": conc_arr[-1].tolist(),
        "equilibrium_ratio": float(rc["k1"] / max(rc["k2"], 1e-30)),
    })


async def _handle_chem_polymer(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.chemistry.polymer")
    if not mod:
        return _err("qulab.labs.chemistry.polymer not found.")
    chain = mod.PolymerChain(
        monomer_mass=args.get("monomer_mass", 150.0),
        n_monomers=args.get("n_monomers", 100),
        temperature=args.get("temperature", 300.0),
        solvent_permittivity=args.get("solvent_permittivity", 78.0),
    )
    r_ee = chain.calculate_end_to_end_distance()
    eps = chain.calculate_dielectric_screening()
    n = args.get("n_monomers", 100)
    return _ok({
        "n_monomers": n,
        "monomer_mass_g_per_mol": args.get("monomer_mass", 150.0),
        "end_to_end_distance": _safe(r_ee),
        "radius_of_gyration_approx": _safe(r_ee) / np.sqrt(6) if not isinstance(r_ee, np.ndarray) else None,
        "dielectric_screening": _safe(eps),
    })


# ── Materials Database handlers ───────────────────────────────────────────────

async def _handle_materials_lookup(args: dict) -> CallToolResult:
    db = _get_materials_db()
    if db is None:
        return _err("Materials database not available. Run from the QuLabInfinite repo root.")
    mat = db.get_material(args["name"])
    if mat is None:
        # Try case-insensitive search
        results = db.search_materials()
        matches = [r for r in results if args["name"].lower() in r.name.lower()]
        if matches:
            mat = matches[0]
        else:
            cats = db.list_categories()
            return _err(f"Material '{args['name']}' not found. Try materials_search or materials_categories.")
    # Serialize the dataclass — keep non-zero / non-default fields
    raw = vars(mat)
    clean = {k: v for k, v in raw.items()
             if v not in (None, 0, 0.0, "", [], 1.0, [0, 14])
             or k in ("name", "category", "density", "youngs_modulus")}
    return _ok(clean)


async def _handle_materials_search(args: dict) -> CallToolResult:
    db = _get_materials_db()
    if db is None:
        return _err("Materials database not available.")
    limit = min(args.get("max_results", 10), 50)
    kwargs: dict = {}
    for k in ("category", "subcategory", "availability"):
        if args.get(k) is not None:
            kwargs[k] = args[k]
    if args.get("text") is not None:
        kwargs["text"] = args["text"]
    for k in ("min_density", "max_density", "min_strength", "max_strength",
              "min_youngs_modulus", "max_cost"):
        if args.get(k) is not None:
            kwargs[k] = args[k]
    results = db.search_materials(**kwargs)[:limit]
    return _ok({
        "count": len(results),
        "materials": [
            {
                "name": r.name,
                "category": r.category,
                "density_kg_m3": r.density,
                "youngs_modulus_GPa": r.youngs_modulus,
                "tensile_strength_MPa": r.tensile_strength,
                "melting_point_K": r.melting_point,
            }
            for r in results
        ],
    })


async def _handle_materials_categories(args: dict) -> CallToolResult:
    db = _get_materials_db()
    if db is None:
        return _err("Materials database not available.")
    cats = db.list_categories()
    stats = db.get_statistics()
    return _ok({"total_materials": db.get_count(), "categories": cats, "stats": _safe(stats)})


async def _handle_materials_design(args: dict) -> CallToolResult:
    # Voigt-Reuss-Hill averages from 3×3 elastic constant matrix (GPa, row-major).
    # For cubic symmetry the 9-element input encodes [C11,C12,C13,C21,C22,C23,C31,C32,C33].
    C = np.array(args["elastic_constants"], dtype=float).reshape(3, 3)
    C11, C12, C13 = C[0]
    C21, C22, C23 = C[1]
    C31, C32, C33 = C[2]
    # Voigt bulk modulus: K_V = (C11+C22+C33 + 2*(C12+C13+C23)) / 9
    K_V = (C11 + C22 + C33 + 2 * (C12 + C13 + C23)) / 9
    # Voigt shear modulus: G_V = ((C11+C22+C33) - (C12+C13+C23) + 3*(C44+C55+C66)) / 15
    # For a 3×3 input we estimate C44 from (C11-C12)/2 (Cauchy relation for cubic)
    C44_est = (C11 - C12) / 2.0
    G_V = ((C11 + C22 + C33) - (C12 + C13 + C23) + 3 * C44_est * 3) / 15
    # Voigt Young's modulus and Poisson's ratio
    E_V = 9 * K_V * G_V / (3 * K_V + G_V) if (3 * K_V + G_V) != 0 else 0
    nu_V = (3 * K_V - 2 * G_V) / (2 * (3 * K_V + G_V)) if (3 * K_V + G_V) != 0 else 0
    return _ok({
        "bulk_modulus_GPa": float(K_V),
        "shear_modulus_GPa": float(G_V),
        "youngs_modulus_GPa": float(E_V),
        "poissons_ratio": float(nu_V),
        "note": "Voigt averages. For full anisotropy provide a 6×6 Cij matrix.",
    })


# ── Nanotech handlers ─────────────────────────────────────────────────────────

async def _handle_nano_qd_bandgap(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.engineering.nanotechnology_lab.nanotech_core")
    if not mod:
        return _err("nanotechnology_lab not found.")
    sim = mod.QuantumDotSimulator()
    result = sim.brus_equation_bandgap(
        radius_nm=args["radius_nm"],
        bulk_bandgap_eV=args["bulk_bandgap_eV"],
        electron_mass_ratio=args.get("electron_mass_ratio", 0.067),
        hole_mass_ratio=args.get("hole_mass_ratio", 0.45),
        dielectric_constant=args.get("dielectric_constant", 12.9),
    )
    return _ok(result)


async def _handle_nano_surface_area(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.engineering.nanotechnology_lab.nanotech_core")
    if not mod:
        return _err("nanotechnology_lab not found.")
    props = mod.NanomaterialProperties()
    result = props.specific_surface_area(
        diameter_nm=args["diameter_nm"],
        density_g_per_cm3=args["density_g_per_cm3"],
        porosity=args.get("porosity", 0.0),
    )
    return _ok(result)


async def _handle_nano_melting(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.engineering.nanotechnology_lab.nanotech_core")
    if not mod:
        return _err("nanotechnology_lab not found.")
    props = mod.NanomaterialProperties()
    result = props.melting_point_depression(
        bulk_melting_K=args["bulk_melting_K"],
        diameter_nm=args["diameter_nm"],
        surface_energy_J_per_m2=args["surface_energy_J_per_m2"],
        density_g_per_cm3=args["density_g_per_cm3"],
        heat_of_fusion_kJ_per_mol=args["heat_of_fusion_kJ_per_mol"],
        molar_mass_g_per_mol=args["molar_mass_g_per_mol"],
    )
    return _ok(result)


async def _handle_nano_ostwald(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.engineering.nanotechnology_lab.nanotech_core")
    if not mod:
        return _err("nanotechnology_lab not found.")
    synth = mod.NanoparticleSynthesis()
    result = synth.ostwald_ripening(
        initial_diameters_nm=np.array(args["initial_diameters_nm"]),
        temperature_K=args["temperature_K"],
        time_hours=args["time_hours"],
        surface_tension=args.get("surface_tension", 1.5),
        diffusion_coefficient=args.get("diffusion_coefficient", 1e-12),
    )
    return _ok(result)


async def _handle_nano_drug_release(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.engineering.nanotechnology_lab.nanotech_core")
    if not mod:
        return _err("nanotechnology_lab not found.")
    dds = mod.DrugDeliverySystem()
    t_max = args.get("t_max_hours", 24)
    time_arr = np.linspace(0, t_max, 100)
    result = dds.korsmeyer_peppas_release(
        time_hours=time_arr,
        drug_loading_mg=args["drug_loading_mg"],
        particle_diameter_nm=args["particle_diameter_nm"],
        release_exponent=args.get("release_exponent", 0.5),
        rate_constant=args.get("rate_constant", 0.1),
    )
    return _ok(result)


# ── Semiconductor handlers ────────────────────────────────────────────────────

async def _handle_semi_mosfet_iv(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.engineering.semiconductor_lab.semiconductor_core")
    if not mod:
        return _err("semiconductor_lab not found.")
    tp = mod.TransistorPhysics()
    n = args.get("n_points", 20)
    vgs_r = args.get("V_gs_range", [0, 3])
    vds_r = args.get("V_ds_range", [0, 3])
    V_gs = np.linspace(vgs_r[0], vgs_r[1], n)
    V_ds = np.linspace(vds_r[0], vds_r[1], n)
    result = tp.mosfet_iv_characteristic(
        V_gs_array=V_gs, V_ds_array=V_ds,
        V_th=args["V_th"],
        mu_n=args.get("mu_n", 450),
        C_ox=args.get("C_ox", 3.45e-7),
        W_L_ratio=args.get("W_L_ratio", 10),
    )
    return _ok(result)


async def _handle_semi_threshold(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.engineering.semiconductor_lab.semiconductor_core")
    if not mod:
        return _err("semiconductor_lab not found.")
    tp = mod.TransistorPhysics()
    result = tp.threshold_voltage_calculation(
        oxide_thickness_nm=args["oxide_thickness_nm"],
        substrate_doping_cm3=args["substrate_doping_cm3"],
        oxide_charge_cm2=args.get("oxide_charge_cm2", 1e10),
        metal_work_function_eV=args.get("metal_work_function_eV", 4.5),
        temperature_K=args.get("temperature_K", 300),
    )
    return _ok(result)


async def _handle_semi_pn_junction(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.engineering.semiconductor_lab.semiconductor_core")
    if not mod:
        return _err("semiconductor_lab not found.")
    bs = mod.BandStructure()
    result = bs.pn_junction_built_in_potential(
        N_a_cm3=args["N_a_cm3"],
        N_d_cm3=args["N_d_cm3"],
        temperature_K=args.get("temperature_K", 300),
    )
    return _ok(result)


async def _handle_semi_quantum_well(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.engineering.semiconductor_lab.semiconductor_core")
    if not mod:
        return _err("semiconductor_lab not found.")
    bs = mod.BandStructure()
    result = bs.quantum_well_energy_levels(
        well_width_nm=args["well_width_nm"],
        barrier_height_eV=args["barrier_height_eV"],
        effective_mass_ratio=args.get("effective_mass_ratio", 0.067),
    )
    return _ok(result)


async def _handle_semi_diffusion(args: dict) -> CallToolResult:
    mod = _import("qulab.labs.engineering.semiconductor_lab.semiconductor_core")
    if not mod:
        return _err("semiconductor_lab not found.")
    da = mod.DopingAnalysis()
    depth_r = args.get("depth_range_um", [0, 1])
    n = args.get("n_points", 50)
    depth = np.linspace(depth_r[0], depth_r[1], n)
    result = da.diffusion_profile(
        depth_um=depth,
        surface_concentration_cm3=args["surface_concentration_cm3"],
        diffusion_time_hours=args["diffusion_time_hours"],
        diffusion_coefficient_cm2_per_s=args.get("diffusion_coefficient_cm2_per_s", 1e-13),
        temperature_K=args.get("temperature_K", 1273),
    )
    return _ok(result)


# ── Handler dispatch table ────────────────────────────────────────────────────

HANDLERS: dict[str, object] = {
    "chem_molecular_energy":     _handle_chem_molecular_energy,
    "chem_lattice_energy":       _handle_chem_lattice_energy,
    "chem_band_gap":             _handle_chem_band_gap,
    "chem_crystal_field":        _handle_chem_crystal_field,
    "chem_redox_potential":      _handle_chem_redox_potential,
    "chem_activation_energy":    _handle_chem_activation_energy,
    "chem_nernst_potential":     _handle_chem_nernst,
    "chem_kinetic_rms_velocity": _handle_chem_kinetic_rms_velocity,
    "chem_carnot_efficiency":    _handle_chem_carnot,
    "chem_catalysis_simulate":   _handle_chem_catalysis,
    "chem_polymer_properties":   _handle_chem_polymer,
    "materials_lookup":          _handle_materials_lookup,
    "materials_search":          _handle_materials_search,
    "materials_categories":      _handle_materials_categories,
    "materials_design":          _handle_materials_design,
    "nano_quantum_dot_bandgap":  _handle_nano_qd_bandgap,
    "nano_surface_area":         _handle_nano_surface_area,
    "nano_melting_point_depression": _handle_nano_melting,
    "nano_ostwald_ripening":     _handle_nano_ostwald,
    "nano_drug_release":         _handle_nano_drug_release,
    "semi_mosfet_iv":            _handle_semi_mosfet_iv,
    "semi_threshold_voltage":    _handle_semi_threshold,
    "semi_pn_junction":          _handle_semi_pn_junction,
    "semi_quantum_well":         _handle_semi_quantum_well,
    "semi_diffusion_profile":    _handle_semi_diffusion,
}
