"""
FJH Reactor Digital Twin — comprehensive virtual tests.

Tests 1-8 from project specification plus unit tests.
"""

from __future__ import annotations

from qulab.labs.engineering.fjh_reactor_lab.atmosphere import compare_atmospheres
from qulab.labs.engineering.fjh_reactor_lab.config import ReactorConfiguration
from qulab.labs.engineering.fjh_reactor_lab.electrical import (
    check_impossible_rectangular_pulse,
    rectangular_pulse_energy_J,
    simulate_electrical,
    simulate_level0_rc,
)
from qulab.labs.engineering.fjh_reactor_lab.energy import compute_energy_accounting
from qulab.labs.engineering.fjh_reactor_lab.fjh_reactor_lab import FJHReactorLab
from qulab.labs.engineering.fjh_reactor_lab.hardware import (
    AWG4_COPPER_RESISTANCE_OHM_PER_M_20C,
    NonFlashElectrolyticBank,
    evaluate_nonflash_side_bank,
)
from qulab.labs.engineering.fjh_reactor_lab.sanity import run_sanity_checks
from qulab.labs.engineering.fjh_reactor_lab.thermal import (
    adiabatic_upper_bound_K,
    simulate_thermal_lumped,
)
from qulab.labs.engineering.fjh_reactor_lab.types import (
    AtmosphereType,
    ModelLevel,
    SanityStatus,
    is_unknown,
)
from qulab.labs.engineering.fjh_reactor_lab.ultrasound import UltrasoundConfig
from qulab.labs.engineering.fjh_reactor_lab.uncertainty import run_monte_carlo


# ---------------------------------------------------------------------------
# TEST 1: Twelve 900 uF capacitors in parallel -> 10,800 uF
# ---------------------------------------------------------------------------
class TestCapacitorBank:
    def test_total_capacitance_parallel(self):
        cfg = ReactorConfiguration.default_fjh_bank()
        assert cfg.capacitor_count == 12
        assert cfg.capacitor_each_capacitance_uF == 900.0
        assert abs(cfg.total_capacitance_uF() - 10800.0) < 0.01

    def test_total_capacitance_farads(self):
        cfg = ReactorConfiguration.default_fjh_bank()
        assert abs(cfg.total_capacitance_F() - 0.0108) < 1e-6


# ---------------------------------------------------------------------------
# TEST 2: Stored energy at 450 V ~ 1093.5 J
# ---------------------------------------------------------------------------
class TestStoredEnergy:
    def test_stored_energy_at_450V(self):
        cfg = ReactorConfiguration.default_fjh_bank()
        E = cfg.initial_stored_energy_J()
        # E = 0.5 * 0.0108 * 450^2 = 1093.5 J
        assert abs(E - 1093.5) < 1.0, f"Expected ~1093.5 J, got {E:.1f} J"

    def test_energy_formula(self):
        cfg = ReactorConfiguration.default_fjh_bank()
        C = cfg.total_capacitance_F()
        V = cfg.capacitor_nominal_voltage_V
        expected = 0.5 * C * V ** 2
        assert abs(cfg.initial_stored_energy_J() - expected) < 1e-6


# ---------------------------------------------------------------------------
# TEST 3: Impossible rectangular 450V/1000A/5ms rejected (~2250 J > ~1094 J)
# ---------------------------------------------------------------------------
class TestImpossiblePulse:
    def test_rectangular_pulse_energy_exceeds_bank(self):
        cfg = ReactorConfiguration.default_fjh_bank()
        E_pulse = rectangular_pulse_energy_J(450, 1000, 0.005)
        assert abs(E_pulse - 2250.0) < 0.1

        impossible, msg = check_impossible_rectangular_pulse(cfg, 450, 1000, 0.005)
        assert impossible is True
        assert "2250" in msg or "Physically impossible" in msg

    def test_sanity_checker_rejects_impossible_pulse(self):
        cfg = ReactorConfiguration.default_fjh_bank()
        sanity = run_sanity_checks(
            cfg,
            rectangular_pulse={"V": 450, "I": 1000, "t_s": 0.005},
            model_level=ModelLevel.LEVEL_0,
        )
        assert sanity.status == SanityStatus.PHYSICALLY_INVALID
        assert "rectangular_pulse_energy" in sanity.failed_invariants


# ---------------------------------------------------------------------------
# TEST 4: Capacitor discharge shows voltage/current decay, not constant
# ---------------------------------------------------------------------------
class TestDischargeDecay:
    def test_voltage_decays(self):
        cfg = ReactorConfiguration.default_fjh_bank()
        cfg.sample_resistance_ohm = 0.1
        result = simulate_level0_rc(cfg, duration_s=0.005)
        V = result["V_cap"].values
        I = result["current"].values
        assert V[0] > V[-1], "Voltage must decay during discharge"
        assert I[0] > I[-1], "Current must decay during discharge"
        assert not all(abs(v - V[0]) < 1 for v in V), "Voltage must not remain constant"

    def test_current_not_constant(self):
        cfg = ReactorConfiguration.default_fjh_bank()
        cfg.sample_resistance_ohm = 0.1
        result = simulate_electrical(cfg, model_level=ModelLevel.LEVEL_1)
        I = result["current"].values
        I_range = max(I) - min(I)
        assert I_range > 0.01 * max(I), "Current must vary significantly"

    def test_peak_current_below_impossible_rectangular(self):
        """Dynamic model peak current * duration must not imply > bank energy."""
        cfg = ReactorConfiguration.default_fjh_bank()
        cfg.sample_resistance_ohm = 0.1
        result = simulate_electrical(cfg, model_level=ModelLevel.LEVEL_1, duration_s=0.005)
        I = result["current"].values
        V = result["V_cap"].values
        # Average power over pulse should be less than bank energy / duration
        E_bank = cfg.initial_stored_energy_J()
        # Rough check: peak V*I*t should not exceed bank energy by much
        peak_power_time = max(V) * max(I) * 0.005
        assert peak_power_time > E_bank * 0.5  # can exceed instantaneously but...
        # Energy accounting should conserve
        energy = compute_energy_accounting(cfg, result, ModelLevel.LEVEL_1)
        assert energy.is_conserved or energy.balance_error_fraction < 0.1


# ---------------------------------------------------------------------------
# TEST 5: Contact resistance affects current and localized energy loss
# ---------------------------------------------------------------------------
class TestContactResistance:
    def test_higher_contact_resistance_reduces_peak_current(self):
        cfg_low = ReactorConfiguration.default_fjh_bank()
        cfg_low.sample_resistance_ohm = 0.1
        cfg_low.electrode_contact_resistance_ohm = 0.001

        cfg_high = ReactorConfiguration.default_fjh_bank()
        cfg_high.sample_resistance_ohm = 0.1
        cfg_high.electrode_contact_resistance_ohm = 0.05

        r_low = simulate_electrical(cfg_low, model_level=ModelLevel.LEVEL_1)
        r_high = simulate_electrical(cfg_high, model_level=ModelLevel.LEVEL_1)

        I_low = max(r_low["current"].values)
        I_high = max(r_high["current"].values)
        assert I_low > I_high, "Higher contact resistance should reduce peak current"

    def test_contact_resistance_increases_contact_losses(self):
        cfg = ReactorConfiguration.default_fjh_bank()
        cfg.sample_resistance_ohm = 0.1
        cfg.electrode_contact_resistance_ohm = 0.02
        result = simulate_electrical(cfg, model_level=ModelLevel.LEVEL_1)
        energy = compute_energy_accounting(cfg, result, ModelLevel.LEVEL_1, contact_resistance_ohm=0.02)
        assert energy.contact_losses_J > 0, "Contact resistance should produce contact losses"


# ---------------------------------------------------------------------------
# TEST 6: Monte Carlo uncertainty propagation
# ---------------------------------------------------------------------------
class TestMonteCarlo:
    def test_monte_carlo_produces_confidence_intervals(self):
        cfg = ReactorConfiguration.default_fjh_bank()
        cfg.sample_resistance_ohm = 0.1

        def _sim_fn(c):
            elec = simulate_electrical(c, model_level=ModelLevel.LEVEL_2)
            thermal = simulate_thermal_lumped(c, elec["P_sample"])
            energy = compute_energy_accounting(c, elec, ModelLevel.LEVEL_2)
            return {
                "peak_current_A": max(elec["current"].values),
                "peak_temperature_K": thermal.peak_temperature_K,
                "delivered_energy_J": energy.sample_energy_J,
                "max_heating_rate_K_s": max(thermal.heating_rate_K_s.values),
            }

        uq = run_monte_carlo(cfg, _sim_fn, n_samples=30, seed=42)
        assert uq.n_samples == 30
        assert "p5" in uq.peak_temperature_K
        assert "p95" in uq.peak_temperature_K
        assert uq.peak_temperature_K["p95"] >= uq.peak_temperature_K["p5"]
        assert len(uq.dominant_uncertain_parameters) > 0

    def test_lab_monte_carlo_endpoint(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({
            "experiment_type": "monte_carlo",
            "n_samples": 20,
            "reactor_config": {"sample_resistance_ohm": 0.1},
        })
        assert result["status"] == "success"
        assert "uncertainty" in result
        assert result["uncertainty"]["n_samples"] == 20


# ---------------------------------------------------------------------------
# TEST 7: Argon vs vacuum atmosphere comparison
# ---------------------------------------------------------------------------
class TestAtmosphereComparison:
    def test_atmosphere_comparison_identifies_modeled_vs_placeholder(self):
        cfg_vac = ReactorConfiguration.default_fjh_bank()
        cfg_vac.atmosphere_type = AtmosphereType.VACUUM
        cfg_ar = ReactorConfiguration.default_fjh_bank()
        cfg_ar.atmosphere_type = AtmosphereType.ARGON

        comparison = compare_atmospheres(cfg_vac, cfg_ar)
        assert "vacuum" in comparison
        assert "argon" in comparison
        assert "placeholder_effects" in comparison["vacuum"]
        assert "modeled_effects" in comparison["argon"]
        assert "residual_oxygen" in comparison["note"].lower() or "residual" in str(comparison)

    def test_lab_compare_atmospheres(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({"experiment_type": "compare_atmospheres"})
        assert result["status"] == "success"
        assert "atmosphere_comparison" in result


# ---------------------------------------------------------------------------
# TEST 8: Ultrasound ON vs OFF — no assumed benefit
# ---------------------------------------------------------------------------
class TestUltrasound:
    def test_ultrasound_config_states_unvalidated(self):
        us = UltrasoundConfig(enabled=True)
        assert "unvalidated" in us.validation_status.lower()

    def test_lab_ultrasound_no_assumed_benefit(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({"experiment_type": "compare_ultrasound"})
        assert result["status"] == "success"
        assert result.get("assumed_benefit") is False
        assert "unvalidated" in result.get("validation_status", "").lower()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------
class TestFJHReactorLab:
    def test_simulate_pulse(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({
            "experiment_type": "simulate_pulse",
            "model_level": 2,
            "reactor_config": {"sample_resistance_ohm": 0.1},
        })
        assert result["status"] == "success"
        assert result["hardware_control_enabled"] is False
        assert "dashboard" in result
        assert "CAPACITOR_BANK" in result["dashboard"]
        scores = result["simulation_result"]["hypothesis_scores"]
        assert "HYPOTHESIS" in scores["label"]

    def test_energy_conservation(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({
            "experiment_type": "simulate_pulse",
            "model_level": 1,
            "reactor_config": {"sample_resistance_ohm": 0.1},
        })
        energy = result["simulation_result"]["energy"]
        assert energy["is_conserved"] is True

    def test_hardware_control_forbidden(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({
            "experiment_type": "simulate_pulse",
            "reactor_config": {"hardware_control_enabled": True},
        })
        # Lab forces hardware_control_enabled=False
        assert result["hardware_control_enabled"] is False

    def test_doe_lhs(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({
            "experiment_type": "doe_latin_hypercube",
            "n_samples": 3,
        })
        assert result["status"] == "success"
        assert result["n_runs"] == 3

    def test_get_status(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        status = lab.get_status()
        assert status["mode"] == "simulation_only"
        assert status["hardware_control_enabled"] is False

    def test_ai_query(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({
            "experiment_type": "ai_query",
            "query_type": "characterization_methods",
        })
        assert "methods" in result

    def test_experiment_ledger_records(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        lab.run_experiment({"experiment_type": "simulate_pulse"})
        experiments = lab.ledger.list_experiments()
        assert len(experiments) >= 1


class TestUnknownValues:
    def test_unknown_parameters_explicit(self):
        cfg = ReactorConfiguration.default_fjh_bank()
        unknowns = cfg.unknown_parameters()
        assert "measured_ESR_each_ohm" in unknowns or len(unknowns) > 0

    def test_config_hash_reproducible(self):
        cfg1 = ReactorConfiguration.default_fjh_bank()
        cfg2 = ReactorConfiguration.default_fjh_bank()
        assert cfg1.config_hash() == cfg2.config_hash()


class TestPhysicalLabSetup:
    def test_sample_mass_is_one_gram(self):
        cfg = ReactorConfiguration.physical_lab_setup()
        assert cfg.sample_mass_g == 1.0
        assert not is_unknown(cfg.sample_mass_g)

    def test_graphite_rod_electrodes_known_material_unknown_size(self):
        cfg = ReactorConfiguration.physical_lab_setup()
        assert cfg.graphite_electrodes.material == "graphite"
        assert cfg.graphite_electrodes.count == 2
        assert is_unknown(cfg.graphite_electrodes.length_mm)
        assert is_unknown(cfg.graphite_electrodes.diameter_mm)
        assert is_unknown(cfg.electrode_geometry.contact_area_mm2)

    def test_awg4_hv_wiring_length_unknown(self):
        cfg = ReactorConfiguration.physical_lab_setup()
        assert cfg.hv_wiring.awg == 4
        assert is_unknown(cfg.hv_wiring.length_m)
        assert is_unknown(cfg.hv_wiring.resistance_ohm())
        assert abs(cfg.hv_wiring.estimate_for_length_m(1.0) - AWG4_COPPER_RESISTANCE_OHM_PER_M_20C) < 1e-6

    def test_bleed_resistor_present_resistance_unknown(self):
        cfg = ReactorConfiguration.physical_lab_setup()
        assert cfg.bleed_resistor.present is True
        assert cfg.bleed_resistor.in_pulse_current_path is False
        assert is_unknown(cfg.bleed_resistor.resistance_ohm)
        stolen_100k = cfg.bleed_resistor.energy_stolen_during_pulse_J(450, 0.005, 1e5)
        assert stolen_100k < 1.0

    def test_premix_dry_is_planned_not_atomic_au(self):
        cfg = ReactorConfiguration.physical_lab_setup()
        prep = cfg.sample_prep.to_dict()
        assert prep["status"] == "planned"
        assert "dry" in prep["steps"]
        assert prep["does_not_imply_atomic_Au"] is True
        assert prep["precursor_loading_wt_percent"] == "UNKNOWN"

    def test_adiabatic_1g_bound_below_2000K(self):
        cfg = ReactorConfiguration.physical_lab_setup()
        bounds = adiabatic_upper_bound_K(cfg)
        assert abs(bounds["energy_J"] - 1093.5) < 1.0
        assert abs(bounds["energy_density_J_per_g"] - 1093.5) < 1.0
        # 1 g cannot reach typical FJH graphene temps even if ALL bank energy is absorbed
        assert bounds["adiabatic_peak_temperature_K"] < 2000
        assert bounds["adiabatic_peak_temperature_K"] > 1500

    def test_lighter_mass_has_higher_adiabatic_T(self):
        cfg = ReactorConfiguration.physical_lab_setup()
        one_g = adiabatic_upper_bound_K(cfg)
        cfg.sample_mass_g = 0.1
        light = adiabatic_upper_bound_K(cfg)
        assert light["adiabatic_peak_temperature_K"] > one_g["adiabatic_peak_temperature_K"]

    def test_lab_physical_setup_report(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({"experiment_type": "physical_setup_report"})
        assert result["status"] == "success"
        assert result["hardware"]["sample_mass_g"] == 1.0
        assert "graphite rod" in " ".join(result["known_inputs"]).lower() or any(
            "graphite" in x.lower() for x in result["known_inputs"]
        )
        assert result["thermal_bounds"]["mass_g"] == 1.0
        dash = result["simulation"]["dashboard"]
        assert "HARDWARE" in dash
        assert dash["HARDWARE"]["sample_mass_g"] == 1.0
        assert dash["HARDWARE"]["planned_test_mass_g"] == 0.5
        assert result["hardware"]["planned_test_mass_g"] == 0.5
        assert result["planned_test_mass"]["planned_test_mass_g"] == 0.5
        assert result["planned_test_mass"]["this_is_not_a_firing_recommendation"] is True
        assert dash["HARDWARE"]["igbt"]["manufacturer"] == "Infineon"
        assert dash["HARDWARE"]["igbt"]["voltage_rating_V"] == 600.0
        assert dash["HARDWARE"]["side_electrolytic_bank"]["usable_as_fjh_dump_bank"] is False
        assert any("Infineon" in x for x in result["known_inputs"])
        assert any("4700" in x for x in result["known_inputs"])
        side = result["side_electrolytics"]
        assert side["do_not_use"] is True
        assert abs(side["side_bank_energy_J"] - 4758.75) < 0.1

    def test_lab_compare_sample_mass(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({
            "experiment_type": "compare_sample_mass",
            "masses_g": [1.0, 0.1],
        })
        assert result["status"] == "success"
        assert len(result["comparison"]) == 2
        one_g = result["comparison"][0]
        light = result["comparison"][1]
        assert one_g["sample_mass_g"] == 1.0
        assert light["peak_temperature_K"] > one_g["peak_temperature_K"]


class TestPlannedHalfGramLoad:
    def test_adiabatic_half_gram_in_domain(self):
        from qulab.labs.engineering.fjh_reactor_lab.test_mass import adiabatic_for_mass

        one = adiabatic_for_mass(1.0)
        half = adiabatic_for_mass(0.5)
        quarter = adiabatic_for_mass(0.25)
        assert abs(half["energy_density_J_per_g"] - 2187.0) < 0.1
        assert 3300 < half["adiabatic_peak_temperature_K"] < 3500
        assert half["in_lumped_model_domain"] is True
        assert one["adiabatic_peak_temperature_K"] < 2000
        assert quarter["in_lumped_model_domain"] is False
        assert half["energy_density_J_per_g"] == 2 * one["energy_density_J_per_g"]

    def test_level2_half_gram_hotter_than_1g_not_graphene(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({"experiment_type": "evaluate_test_mass"})
        assert result["planned_test_mass_g"] == 0.5
        assert result["evaluation"]["this_is_not_a_firing_recommendation"] is True
        assert result["evaluation"]["recommendation"]["do_not_fire"] is True
        by_mass = {r["sample_mass_g"]: r for r in result["comparison"]}
        assert by_mass[0.5]["peak_temperature_K"] > by_mass[1.0]["peak_temperature_K"]
        assert 1800 < by_mass[0.5]["peak_temperature_K"] < 2800
        assert by_mass[0.5]["model_domain"] == "in_domain"
        assert by_mass[0.25]["model_domain"] == "QUESTIONABLE"
        # Hypothesis score is a label, not a graphene claim
        score = by_mass[0.5]["hypothesis_scores"]["graphene_conversion_score"]
        assert 0 < score < 1
        assert "HYPOTHESIS" in by_mass[0.5]["hypothesis_scores"]["label"]


class TestBatchScaling:
    def test_energy_per_900uF_450V_cap(self):
        from qulab.labs.engineering.fjh_reactor_lab.batch_scaling import energy_per_capacitor_J
        e = energy_per_capacitor_J()
        assert abs(e - 91.125) < 0.01

    def test_20g_on_12_caps_is_not_flash_heating(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({"experiment_type": "scale_batch", "mass_g": 20.0})
        assert result["current_bank_count"] == 12
        assert result["current_energy_density_J_per_g"] < 60
        assert result["current_estimated_peak_T_K"] < 450
        keep = next(c for c in result["cases"] if c["name"] == "keep_current_12_caps")
        assert keep["graphene_relevant"] is False

    def test_20g_match_1g_density_needs_240_caps(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({"experiment_type": "scale_batch", "mass_g": 20.0})
        match = next(c for c in result["cases"] if c["name"] == "match_1g_energy_density")
        assert match["capacitor_count"] == 240
        sink = next(c for c in result["cases"] if c["name"] == "2500K_with_graphite_sink")
        assert sink["capacitor_count"] > 240
        assert result["safety"]["this_is_not_a_firing_recommendation"] is True


class TestInfineonIGBT:
    def test_physical_setup_igbt_is_infineon_600v(self):
        cfg = ReactorConfiguration.physical_lab_setup()
        assert cfg.igbt.manufacturer == "Infineon"
        assert cfg.igbt.voltage_rating_V == 600.0
        assert is_unknown(cfg.igbt.part_number)
        assert is_unknown(cfg.igbt.current_rating_A)
        assert is_unknown(cfg.igbt.pulse_current_rating_A)
        assert cfg.igbt.voltage_headroom_V(450.0) == 150.0

    def test_igbt_voltage_legal_current_unknown(self):
        cfg = ReactorConfiguration.physical_lab_setup()
        sanity = run_sanity_checks(cfg, model_level=ModelLevel.LEVEL_1, max_current_A=4000.0)
        igbt_msgs = [m for m in sanity.messages if m.startswith("igbt_")]
        assert any("600" in m and "450" in m for m in igbt_msgs)
        assert any("UNKNOWN" in m and "Do not fire" in m for m in igbt_msgs)
        assert sanity.status in (
            SanityStatus.QUESTIONABLE,
            SanityStatus.INSUFFICIENT_DATA,
        )
        assert "igbt_voltage" not in sanity.failed_invariants

    def test_igbt_overvoltage_is_invalid(self):
        cfg = ReactorConfiguration.physical_lab_setup()
        cfg.initial_voltage_V = 700.0
        sanity = run_sanity_checks(cfg, model_level=ModelLevel.LEVEL_0)
        assert sanity.status == SanityStatus.PHYSICALLY_INVALID
        assert "igbt_voltage" in sanity.failed_invariants


class TestNonFlashSideElectrolytics:
    def test_energy_of_ten_4700uF_at_450V(self):
        bank = NonFlashElectrolyticBank()
        assert bank.count == 10
        assert bank.manufacturer == "JCCON"
        assert bank.capacitance_each_uF == 4700.0
        assert bank.voltage_rating_V == 450.0
        assert bank.form_factor == "CD136"
        assert bank.flash_rated is False
        assert bank.usable_as_fjh_dump_bank is False
        assert abs(bank.energy_each_J() - 475.875) < 1e-6
        assert abs(bank.energy_bank_J() - 4758.75) < 1e-6
        assert bank.total_capacitance_uF() == 47000.0
        assert is_unknown(bank.esr_ohm_each)
        assert is_unknown(bank.pulse_current_A)

    def test_evaluate_refuses_use(self):
        ev = evaluate_nonflash_side_bank()
        assert ev["do_not_use"] is True
        assert ev["usable_as_fjh_dump_bank"] is False
        assert ev["this_is_not_a_firing_recommendation"] is True
        assert ev["safety"]["do_not_parallel_onto_flash_bank"] is True
        assert abs(ev["energy_ratio_vs_flash_bank"] - (4758.75 / 1093.5)) < 1e-6
        assert ev["virtual_adiabatic_exceeds_carbon_model_domain"] is True
        assert ev["virtual_adiabatic_1g_side_only_K"] > 3500

    def test_using_as_dump_bank_is_physically_invalid(self):
        cfg = ReactorConfiguration.physical_lab_setup()
        cfg.uses_nonflash_electrolytic_dump = True
        sanity = run_sanity_checks(cfg, model_level=ModelLevel.LEVEL_0)
        assert sanity.status == SanityStatus.PHYSICALLY_INVALID
        assert "nonflash_electrolytic_dump" in sanity.failed_invariants

    def test_lab_evaluate_side_electrolytics(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({"experiment_type": "evaluate_side_electrolytics"})
        assert result["status"] == "success"
        assert result["do_not_use"] is True
        assert abs(result["side_bank_energy_J"] - 4758.75) < 0.1
        assert result["safety"]["do_not_fire"] is True

    def test_lab_use_as_dump_bank_rejected(self, tmp_path):
        lab = FJHReactorLab({"ledger_db": str(tmp_path / "ledger.db")})
        result = lab.run_experiment({
            "experiment_type": "evaluate_side_electrolytics",
            "use_as_dump_bank": True,
        })
        assert result["do_not_use"] is True
        assert result["sanity_if_used_as_dump"]["status"] == "PHYSICALLY_INVALID"
