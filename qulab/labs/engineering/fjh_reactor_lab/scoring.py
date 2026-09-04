"""
Material outcome hypothesis scoring layer.

IMPORTANT: All outputs are HYPOTHESIS SCORES until experimentally calibrated.
Never display "X% single atom gold" without validated empirical model.
"""

from __future__ import annotations

from .material import MaterialHypothesis
from .thermal import ThermalResult
from .types import HypothesisScores


def compute_hypothesis_scores(
    thermal: ThermalResult | None,
    material: MaterialHypothesis,
    peak_current_A: float,
    delivered_energy_J: float,
    atmosphere_type: str = "argon",
    pressure_Pa: float | None = None,
    precursor_uniformity: float | None = None,
    ultrasound_enabled: bool = False,
) -> HypothesisScores:
    """
    Preliminary qualitative/probabilistic outcome layer.

    Scores are 0-1 where higher = more likely/riskier depending on metric.
    These are HYPOTHESIS SCORES, not validated predictions.
    """
    peak_T = thermal.peak_temperature_K if thermal else 298.15
    time_above_1000 = thermal.time_above_thresholds.get("1000K", 0.0) if thermal else 0.0
    time_above_2000 = thermal.time_above_thresholds.get("2000K", 0.0) if thermal else 0.0
    max_heating = 0.0
    if thermal and thermal.heating_rate_K_s.values:
        max_heating = max(thermal.heating_rate_K_s.values)

    # Carbon conversion — higher T and longer high-T exposure favor graphitization
    graphene_score = min(1.0, max(0.0, (peak_T - 800) / 2200))
    graphene_score *= min(1.0, time_above_1000 / 0.001 + 0.1)

    # Au single-atom retention — moderate T, fast quench favor; overheating hurts
    au_retention = 0.5
    if peak_T < 1500:
        au_retention = 0.6 + 0.2 * min(1.0, delivered_energy_J / 500)
    elif peak_T > 2500:
        au_retention = max(0.1, 0.5 - (peak_T - 2500) / 3000)
    if time_above_2000 > 0.001:
        au_retention *= 0.5

    # Aggregation risks increase with temperature and energy
    au_cluster_risk = min(1.0, max(0.0, (peak_T - 1000) / 2000))
    au_nanoparticle_risk = min(1.0, max(0.0, (peak_T - 1200) / 1800))
    au_loss_risk = min(1.0, max(0.0, (peak_T - 1800) / 1500))
    if atmosphere_type == "vacuum":
        au_loss_risk = min(1.0, au_loss_risk * 1.3)

    carbon_damage = min(1.0, max(0.0, (peak_T - 2000) / 2000 + max_heating / 1e6))

    # Precursor uniformity modulates scores
    if precursor_uniformity is not None:
        au_retention *= 0.5 + 0.5 * precursor_uniformity
        au_cluster_risk *= 1.5 - 0.5 * precursor_uniformity

    # Ultrasound: no validated benefit assumed
    if ultrasound_enabled:
        pass  # scores unchanged; effect unvalidated

    return HypothesisScores(
        graphene_conversion_score=round(graphene_score, 4),
        au_single_atom_retention_score=round(au_retention, 4),
        au_cluster_risk=round(au_cluster_risk, 4),
        au_nanoparticle_risk=round(au_nanoparticle_risk, 4),
        au_loss_risk=round(au_loss_risk, 4),
        carbon_damage_risk=round(carbon_damage, 4),
    )
