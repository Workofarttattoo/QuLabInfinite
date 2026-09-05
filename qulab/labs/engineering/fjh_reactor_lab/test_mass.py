"""
Virtual test-mass comparison for the 12×900 µF / 450 V flash bank.

0.5 g is the planned virtual test load: better energy density than 1 g,
still inside the lumped-model domain. This is not a firing plan.
"""

from __future__ import annotations

from typing import Any

from .thermal import VULCAN_XC72_THERMAL, _effective_cp

PLANNED_TEST_MASS_G = 0.5
PREVIOUSLY_STATED_MASS_G = 1.0
BANK_ENERGY_J = 1093.5
T0_K = 298.15
MODEL_DOMAIN_LIMIT_K = 3500.0
LITERATURE_FG_J_PER_G = 7200.0
LITERATURE_FG_T_K = 3000.0


def adiabatic_for_mass(
    mass_g: float,
    energy_J: float = BANK_ENERGY_J,
    cp_J_kg_K: float | None = None,
    t0_K: float = T0_K,
) -> dict[str, float]:
    cp = _effective_cp(VULCAN_XC72_THERMAL) if cp_J_kg_K is None else cp_J_kg_K
    mass_kg = mass_g / 1000.0
    j_per_g = energy_J / mass_g if mass_g else 0.0
    delta_T = energy_J / (mass_kg * cp) if mass_kg * cp > 0 else 0.0
    return {
        "mass_g": mass_g,
        "energy_J": energy_J,
        "energy_density_J_per_g": j_per_g,
        "specific_heat_J_kg_K": cp,
        "adiabatic_peak_temperature_K": t0_K + delta_T,
        "delta_T_K": delta_T,
        "in_lumped_model_domain": (t0_K + delta_T) <= MODEL_DOMAIN_LIMIT_K,
        "vs_literature_fg_energy_fraction": (
            j_per_g / LITERATURE_FG_J_PER_G if LITERATURE_FG_J_PER_G else 0.0
        ),
    }


def evaluate_planned_test_mass(
    energy_J: float = BANK_ENERGY_J,
    masses_g: tuple[float, ...] = (1.0, 0.5, 0.25),
    level2_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Compare candidate test masses on the flash bank.

    0.5 g is the least-bad virtual test load on 1093.5 J. It is not graphene
    and it is not authorization to fire.
    """
    adiabatic = [adiabatic_for_mass(m, energy_J) for m in masses_g]
    half = adiabatic_for_mass(PLANNED_TEST_MASS_G, energy_J)
    one = adiabatic_for_mass(1.0, energy_J)
    return {
        "planned_test_mass_g": PLANNED_TEST_MASS_G,
        "previously_stated_mass_g": PREVIOUSLY_STATED_MASS_G,
        "this_is_not_a_firing_recommendation": True,
        "hardware_control_enabled": False,
        "why_half_gram_is_the_virtual_choice": [
            (
                f"0.5 g doubles adiabatic energy density vs 1 g "
                f"({half['energy_density_J_per_g']:.1f} J/g vs "
                f"{one['energy_density_J_per_g']:.1f} J/g) on the same 1093.5 J bank."
            ),
            (
                f"Adiabatic ceiling at 0.5 g is ~{half['adiabatic_peak_temperature_K']:.0f} K, "
                "still inside the lumped-model domain (<3500 K). 0.25 g is not."
            ),
            (
                "LEVEL 2 with a hypothesized 30% graphite-rod sink peaks near 2100 K "
                "at 0.5 g vs ~1200 K at 1 g. That is hotter, not flash-graphene."
            ),
            (
                f"Literature flash-graphene energy is order {LITERATURE_FG_J_PER_G:.0f} J/g "
                f"and T > {LITERATURE_FG_T_K:.0f} K. 0.5 g adiabatic is only "
                f"~{half['vs_literature_fg_energy_fraction']*100:.0f}% of that energy density."
            ),
        ],
        "why_half_gram_is_not_a_fire_plan": [
            "The long graphite rods do not shrink when the powder mass is halved. "
            "A fixed 30% sink overstates the benefit of going lighter; the real rod "
            "fraction may rise, so peak T may stay closer to the 1 g result.",
            "Peak current is set by voltage and path resistance, not sample mass. "
            "The Infineon 600 V IGBT still has UNKNOWN current. 0.5 g does not make the switch safer.",
            "Hypothesis scores are labels, not composition. ~2100 K is not graphene.",
            "Residual water/chloride in a dried Vulcan + HAuCl4 premix is still UNKNOWN.",
            "Do not add the non-flash JCCON 10×4700 µF cans to chase more energy.",
        ],
        "adiabatic": adiabatic,
        "level2_rows": level2_rows,
        "recommendation": {
            "virtual_test_mass_g": PLANNED_TEST_MASS_G,
            "do_not_use_1g_if_the_goal_is_graphene_relevant_T": True,
            "do_not_go_to_0.25g_in_this_lumped_model": True,
            "do_not_fire": True,
        },
    }
