"""
Virtual energy budget for an e-waste / gold-recovery pivot.

Literature FJH urban mining (Deng/Tour, Nat. Commun. 2021) does not
"keep gold and vaporize everything else." It heats powdered PCB + carbon
black to ~3400 K and evaporates metals — including Au, Cu, Pb, Hg —
into a vacuum cold trap. Plastics carbonize.

This bank is 1093.5 J. Literature energy is ~939 kWh/ton ≈ 3380 J/g.
This is not a firing plan. Ground computer parts plus a capacitor dump
without a sealed trap vents toxic metal and halogenated fumes.
"""

from __future__ import annotations

from typing import Any

from .test_mass import BANK_ENERGY_J

# Deng, Luong, Tour et al., Nat. Commun. 12, 5794 (2021).
LITERATURE_FJH_EWASTE_J_PER_G = 3380.4  # 939 kWh / ton
LITERATURE_FJH_EWASTE_T_K = 3400.0
LITERATURE_BANK_CAPACITANCE_F = 0.060
LITERATURE_DOI = "https://doi.org/10.1038/s41467-021-26038-9"

# Order-of-magnitude thermal properties. Not measurements of this grind.
# Polymers decompose before a true boiling point; "vaporize plastics" ≈ pyrolysis.
PYROLYSIS_INPUT_J_PER_G_PLASTIC = 2000.0  # heat + endothermic decomposition, order of magnitude
CU_BOIL_K = 2835.0
AU_BOIL_K = 3243.0
AU_MELT_K = 1337.0
CU_HVAP_J_PER_G = 4728.0  # 300.4 kJ/mol / 63.55 g/mol
AU_HVAP_J_PER_G = 1645.0  # 324 kJ/mol / 196.97 g/mol
CU_CP_J_G_K = 0.385
AU_CP_J_G_K = 0.129

# PCB gold is highly variable. Paper: several–tens of ppm on their board.
# High-grade PCB scrap is often cited ~200–500 g/t (200–500 ppm).
AU_PPM_LOW = 10.0
AU_PPM_HIGH_GRADE = 300.0

# Typical PCB mass fractions — literature-order, UNKNOWN for operator grind.
PCB_PLASTIC_FRACTION = 0.30
PCB_COPPER_FRACTION = 0.18


def literature_mass_this_bank_can_match_g(bank_energy_J: float = BANK_ENERGY_J) -> float:
    return bank_energy_J / LITERATURE_FJH_EWASTE_J_PER_G


def gold_in_feed_g(mass_g: float, au_ppm: float) -> float:
    return mass_g * (au_ppm * 1e-6)


def energy_to_pyrolyze_plastics_J(feed_mass_g: float, plastic_fraction: float = PCB_PLASTIC_FRACTION) -> float:
    return feed_mass_g * plastic_fraction * PYROLYSIS_INPUT_J_PER_G_PLASTIC


def energy_to_vaporize_copper_J(feed_mass_g: float, copper_fraction: float = PCB_COPPER_FRACTION, t0_K: float = 298.15) -> dict[str, float]:
    m_cu = feed_mass_g * copper_fraction
    heat = m_cu * CU_CP_J_G_K * (CU_BOIL_K - t0_K)
    vap = m_cu * CU_HVAP_J_PER_G
    return {
        "copper_mass_g": m_cu,
        "heat_to_boil_J": heat,
        "latent_vaporization_J": vap,
        "total_J": heat + vap,
    }


def energy_to_vaporize_gold_J(gold_mass_g: float, t0_K: float = 298.15) -> dict[str, float]:
    heat = gold_mass_g * AU_CP_J_G_K * (AU_BOIL_K - t0_K)
    vap = gold_mass_g * AU_HVAP_J_PER_G
    return {
        "gold_mass_g": gold_mass_g,
        "heat_to_boil_J": heat,
        "latent_vaporization_J": vap,
        "total_J": heat + vap,
    }


def evaluate_ewaste_gold_pivot(bank_energy_J: float = BANK_ENERGY_J) -> dict[str, Any]:
    """
    Virtual verdict for recycling ground computer parts on this bank.

    Viable on this hardware: no.
    """
    match_g = literature_mass_this_bank_can_match_g(bank_energy_J)
    plastic_J_1g = energy_to_pyrolyze_plastics_J(1.0)
    cu_1g = energy_to_vaporize_copper_J(1.0)
    au_low = gold_in_feed_g(match_g, AU_PPM_LOW)
    au_high = gold_in_feed_g(match_g, AU_PPM_HIGH_GRADE)
    shots_for_1g_au_high = (
        1.0 / (gold_in_feed_g(match_g, AU_PPM_HIGH_GRADE) * 0.60)
        if match_g > 0
        else float("inf")
    )

    return {
        "job": "Recover gold from ground computer parts by FJH",
        "viable_on_this_bank": False,
        "verdict": "NOT_VIABLE",
        "this_is_not_a_firing_recommendation": True,
        "hardware_control_enabled": False,
        "do_not_fire_e_waste": True,
        "why_not_viable": [
            (
                "Published FJH urban mining uses ~3390 J/g and ~3400 K, a 60 mF bank, "
                "quartz tube, vacuum, liquid-nitrogen cold trap, ~30 wt% carbon black, "
                "and halide salts. This machine is 1093.5 J and 10.8 mF with no trap."
            ),
            (
                "The published process evaporates metals including gold, then condenses "
                "them. It does not leave a gold button while vaporizing everything else. "
                "Copper boils at 2835 K, gold at 3243 K — you cannot boil Cu away and keep Au."
            ),
            (
                f"At the literature 3380 J/g, this bank matches only ~{match_g:.2f} g of "
                "feed per shot. Gold in PCB is tens to a few hundred ppm, so one shot "
                "is micrograms of Au, not a recoverable bead."
            ),
            (
                "Ground computer parts are mostly insulating plastic and glass fiber. "
                "Without a conductive additive they will not form an FJH current path."
            ),
            (
                "FR4, PVC, brominated flame retardants, solder, and heavy metals make "
                "an untrapped flash a toxic-fume event (HCl, HBr, dioxins, Pb, Cd, Hg). "
                "Do not dump the bank into a grind pile."
            ),
        ],
        "literature": {
            "citation": (
                "Deng, Luong, Tour et al., Urban mining by flash Joule heating, "
                "Nat. Commun. 12, 5794 (2021)"
            ),
            "doi": LITERATURE_DOI,
            "energy_J_per_g": LITERATURE_FJH_EWASTE_J_PER_G,
            "energy_kWh_per_ton": 939.0,
            "peak_T_K": LITERATURE_FJH_EWASTE_T_K,
            "bank_capacitance_F": LITERATURE_BANK_CAPACITANCE_F,
            "this_bank_capacitance_F": 0.0108,
            "this_bank_energy_J": bank_energy_J,
            "au_evaporative_yield_without_halide": "~3%",
            "au_evaporative_yield_with_NaI": ">60%",
            "condensate_is_mostly_copper": True,
            "plastics_are_carbonized_not_cleanly_vaporized": True,
        },
        "energy_to_remove_non_gold": {
            "note": (
                "These are order-of-magnitude VIRTUAL numbers for 1 g of typical PCB "
                "grind. Operator grind composition is UNKNOWN. Plastics pyrolyze; "
                "they do not boil as clean vapor. Vaporizing copper consumes more "
                "energy than this entire bank."
            ),
            "pyrolyze_plastics_in_1g_pcb_J": plastic_J_1g,
            "vaporize_copper_in_1g_pcb": cu_1g,
            "vaporize_10ug_gold_J": energy_to_vaporize_gold_J(10e-6)["total_J"],
            "cannot_selectively_vaporize_cu_keep_au": True,
            "cu_boils_K": CU_BOIL_K,
            "au_boils_K": AU_BOIL_K,
            "bank_covers_plastic_pyrolysis_of_about_g": (
                bank_energy_J / (PCB_PLASTIC_FRACTION * PYROLYSIS_INPUT_J_PER_G_PLASTIC)
            ),
            "bank_vs_cu_vaporization_in_1g": bank_energy_J / cu_1g["total_J"],
        },
        "gold_yield_virtual": {
            "literature_matched_feed_mass_g": match_g,
            "au_mass_g_if_10_ppm": au_low,
            "au_mass_g_if_300_ppm": au_high,
            "au_mass_ug_if_10_ppm": au_low * 1e6,
            "au_mass_ug_if_300_ppm": au_high * 1e6,
            "shots_to_1g_au_at_300ppm_60pct_recovery": shots_for_1g_au_high,
            "note": (
                "Even the optimistic 300 ppm case is ~0.1 mg Au per literature-matched "
                "shot. Getting 1 g of gold would take thousands of shots and kilograms "
                "of boards — and this bank does not match the literature energy or trap."
            ),
        },
        "what_would_be_required_and_is_missing": [
            "60 mF-class flash-rated bank or equivalent energy at safe pulse rating",
            "Conductive additive (~30 wt% carbon black) mixed with powdered PCB",
            "Quartz tube, vacuum, and a real cold trap (literature used liquid N2)",
            "Halide chemistry if evaporative Au yield matters (NaI in the paper)",
            "Downstream refining: condensate is mostly copper alloy, not bullion",
            "Industrial exhaust / legal e-waste handling — not a kitchen capacitor dump",
        ],
        "safety": {
            "do_not_fire": True,
            "do_not_use_jccon_nonflash_cans": True,
            "untrapped_flash_vents_toxic_metals_and_halogens": True,
        },
    }
