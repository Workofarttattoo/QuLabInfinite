"""
Tool pricing: 1 credit = $0.01
"""

# Tool name → credit cost
TOOL_COSTS: dict[str, int] = {
    # Tier 1 — 1 credit
    "materials_categories": 1,
    "materials_lookup": 1,
    "chem_carnot_efficiency": 1,
    "chem_kinetic_rms_velocity": 1,
    "chem_nernst_potential": 1,
    "thermo_equilibrium_constant": 1,
    "thermo_clausius_clapeyron": 1,
    # Tier 2 — 2 credits
    "materials_search": 2,
    "materials_design": 2,
    "materials_recommend": 2,
    "chem_activation_energy": 2,
    "chem_band_gap": 2,
    "chem_lattice_energy": 2,
    "chem_crystal_field": 2,
    "chem_redox_potential": 2,
    "chem_polymer_properties": 2,
    "pharma_pk_model": 2,
    "pharma_emax_model": 2,
    "genomics_align": 2,
    "nano_surface_area": 2,
    "nano_melting_point_depression": 2,
    "astro_cepheid_luminosity": 2,
    "astro_schwarzschild": 2,
    "particle_breit_wigner": 2,
    # Tier 3 — 5 credits
    "quantum_bell_state": 5,
    "quantum_grovers_search": 5,
    "quantum_teleportation": 5,
    "particle_cross_section": 5,
    "particle_decay_rate": 5,
    "astro_lane_emden": 5,
    "genomics_call_variants": 5,
    "chem_molecular_energy": 5,
    "chem_catalysis_simulate": 5,
    "nano_quantum_dot_bandgap": 5,
    "nano_ostwald_ripening": 5,
    "nano_drug_release": 5,
    "semi_mosfet_iv": 5,
    "semi_threshold_voltage": 5,
    "semi_pn_junction": 5,
    "semi_quantum_well": 5,
    "semi_diffusion_profile": 5,
}

# Default cost for unknown tools
DEFAULT_TOOL_COST: int = 2

# Free methods (no credits charged)
FREE_METHODS: set[str] = {"initialize", "tools/list", "ping", "notifications/initialized"}


def get_tool_cost(tool_name: str) -> int:
    """Return credit cost for a given tool name."""
    return TOOL_COSTS.get(tool_name, DEFAULT_TOOL_COST)


# Credit bundle definitions: 1 credit = $0.01
CREDIT_BUNDLES: list[dict] = [
    {
        "index": 0,
        "label": "Starter",
        "credits": 1000,
        "price_cents": 1000,
        "price_usd": 10.00,
        "description": "1,000 credits for $10.00",
    },
    {
        "index": 1,
        "label": "Standard",
        "credits": 5000,
        "price_cents": 4000,
        "price_usd": 40.00,
        "description": "5,000 credits for $40.00 (20% off)",
        "savings": "20% off",
    },
    {
        "index": 2,
        "label": "Pro",
        "credits": 20000,
        "price_cents": 12000,
        "price_usd": 120.00,
        "description": "20,000 credits for $120.00 (40% off)",
        "savings": "40% off",
    },
]
