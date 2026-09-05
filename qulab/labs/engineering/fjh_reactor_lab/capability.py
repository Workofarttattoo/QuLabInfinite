"""
Capability verdict: is this reactor bank sized for the intended FJH job?

Job assumed: Tour-style flash graphene / high-T carbon conversion
(literature order 7 kJ/g and T > 3000 K).

This is a virtual energy/temperature comparison. It is not a build list
and not a firing authorization.
"""

from __future__ import annotations

import math
from typing import Any

from .test_mass import (
    BANK_ENERGY_J,
    LITERATURE_FG_J_PER_G,
    LITERATURE_FG_T_K,
    PLANNED_TEST_MASS_G,
    T0_K,
    adiabatic_for_mass,
)
from .thermal import VULCAN_XC72_THERMAL, _effective_cp

ENERGY_PER_900UF_450V_CAP_J = 91.125
CURRENT_CAP_COUNT = 12
TOUR_NATURE_BANK_J = 0.5 * 0.22 * 400.0 ** 2  # ~17600 J, Luong/Tour Nature 2020
# Last LEVEL 2 lumped-model peaks on the physical setup (hypothesized 30% rod sink).
LEVEL2_PEAK_T_K = {
    1.0: 1201.0,
    0.5: 2104.0,
}


def capacitors_for_energy_J(energy_J: float, e_each_J: float = ENERGY_PER_900UF_450V_CAP_J) -> int:
    if e_each_J <= 0:
        raise ValueError("energy per capacitor must be positive")
    return int(math.ceil(energy_J / e_each_J))


def machine_capability_verdict(
    bank_energy_J: float = BANK_ENERGY_J,
    planned_mass_g: float = PLANNED_TEST_MASS_G,
    stated_mass_g: float = 1.0,
    electrode_sink_fraction: float = 0.30,
) -> dict[str, Any]:
    """
    Compare the 12×900 µF / 450 V machine to the flash-graphene job.

    Verdict: UNDERPOWERED for that job. 0.5 g is less underpowered than 1 g.
    """
    cp = _effective_cp(VULCAN_XC72_THERMAL)
    one = adiabatic_for_mass(stated_mass_g, bank_energy_J)
    half = adiabatic_for_mass(planned_mass_g, bank_energy_J)
    e_for_lit_half = LITERATURE_FG_J_PER_G * planned_mass_g
    e_for_lit_one = LITERATURE_FG_J_PER_G * stated_mass_g
    sample_fraction = max(1.0 - electrode_sink_fraction, 0.05)
    e_for_3000K_half_with_sink = (
        (planned_mass_g / 1000.0) * cp * (LITERATURE_FG_T_K - T0_K) / sample_fraction
    )
    caps_for_lit_half = capacitors_for_energy_J(e_for_lit_half)
    caps_for_3000K_half = capacitors_for_energy_J(e_for_3000K_half_with_sink)

    return {
        "job": (
            "Tour-style flash graphene / high-T FJH "
            f"(~{LITERATURE_FG_J_PER_G:.0f} J/g, T > {LITERATURE_FG_T_K:.0f} K)"
        ),
        "machine": {
            "capacitor_count": CURRENT_CAP_COUNT,
            "capacitance_each_uF": 900.0,
            "voltage_V": 450.0,
            "stored_energy_J": bank_energy_J,
            "igbt": "Infineon 600 V, current UNKNOWN",
            "electrodes": "two long thin graphite rods (heat sink)",
            "side_inventory": "10× JCCON 4700 µF / 450 V, not flash-rated, do not use",
        },
        "verdict": "UNDERPOWERED",
        "underpowered_for_this_job": True,
        "this_is_not_a_firing_recommendation": True,
        "hardware_control_enabled": False,
        "at_1g": {
            "adiabatic_T_K": one["adiabatic_peak_temperature_K"],
            "adiabatic_J_per_g": one["energy_density_J_per_g"],
            "level2_peak_T_K_hypothesis": LEVEL2_PEAK_T_K[1.0],
            "fraction_of_literature_energy": one["vs_literature_fg_energy_fraction"],
            "graphene_relevant": False,
        },
        "at_0.5g_planned_test": {
            "adiabatic_T_K": half["adiabatic_peak_temperature_K"],
            "adiabatic_J_per_g": half["energy_density_J_per_g"],
            "level2_peak_T_K_hypothesis": LEVEL2_PEAK_T_K[0.5],
            "fraction_of_literature_energy": half["vs_literature_fg_energy_fraction"],
            "graphene_relevant": False,
            "note": (
                "Adiabatic 0.5 g can exceed 3000 K if every joule stays in the powder. "
                "LEVEL 2 with a hypothesized rod sink does not (~2100 K). "
                "Energy density is still only ~30% of literature flash graphene."
            ),
        },
        "vs_tour_nature_2020_bank": {
            "tour_approx_stored_energy_J": TOUR_NATURE_BANK_J,
            "this_machine_energy_J": bank_energy_J,
            "this_machine_fraction": bank_energy_J / TOUR_NATURE_BANK_J,
            "note": (
                "Luong/Tour Nature 2020 used ~0.22 F near 400 V (~17.6 kJ). "
                "This 12×900 µF bank is ~1.1 kJ, about 16× less stored energy."
            ),
        },
        "limiters": [
            "Stored energy: 1093.5 J is too small for 1 g and still short of literature J/g at 0.5 g.",
            "Graphite rods: fixed heat capacity steals a larger fraction as powder mass drops.",
            "Infineon IGBT: 600 V is voltage-legal; pulse current is UNKNOWN and is a second limiter.",
            "Non-flash JCCON 10×4700 µF cans add joules on paper and are not a legal dump path.",
        ],
        "virtual_energy_budget_only": {
            "identical_900uF_450V_caps_to_match_7kJ_per_g_at_0.5g": caps_for_lit_half,
            "identical_900uF_450V_caps_to_match_7kJ_per_g_at_1g": capacitors_for_energy_J(e_for_lit_one),
            "identical_900uF_450V_caps_for_3000K_at_0.5g_with_30pct_rod_sink": caps_for_3000K_half,
            "energy_J_for_3000K_at_0.5g_with_30pct_rod_sink": e_for_3000K_half_with_sink,
            "do_not_treat_as_shopping_list": True,
            "note": (
                "These counts assume the same flash-rated 900 µF / 450 V parts. "
                "A larger bank is a larger stored-energy hazard. The current IGBT, "
                "4 AWG lengths, and bleeder are not automatically valid at that scale."
            ),
        },
        "what_this_machine_can_do": [
            "Virtual DOE and energy accounting on a real 12×900 µF geometry.",
            "Learn RC/RLC discharge, contact-loss, and rod-sink sensitivity.",
            "Not Tour-style flash graphene at 0.5–1 g.",
            "Not a single-atom-gold claim from a pulse that never ran.",
        ],
    }
