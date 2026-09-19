"""Ground-truth tests for the FJH reactor and the repaired electrical helpers."""

from __future__ import annotations

import pytest

from qulab.labs.engineering.electrical import (
    Battery,
    Capacitor,
    ElectricalLab,
    Resistor,
    calculate_power,
    calculate_resistance,
    capacitor_energy,
    voltage_from_ir,
)
from qulab.labs.engineering.fjh_reactor import (
    FJHBank,
    FJHPath,
    FJHSample,
    capacitance_for_target,
    compare_lead_resistances,
    electrode,
    electrode_heating,
    electrode_resistance_ohm,
    fire,
    series_energy_split,
    si_c_pellet,
)


def test_electrical_module_imports():
    Battery(voltage=12.0)  # capacity now has a default
    assert abs(calculate_resistance(12.0, 3.0) - 4.0) < 1e-12
    assert abs(voltage_from_ir(1000.0, 2.0) - 2000.0) < 1e-12
    assert abs(calculate_power(4.0, 3.0) - 36.0) < 1e-12
    assert abs(capacitor_energy(0.25, 150.0) - 2812.5) < 1e-9


def test_simulate_circuit_uses_ir_and_half_cv2():
    lab = ElectricalLab()
    lab.add_component(Resistor("R1", 5.0, 2.0, resistance=1000.0))
    lab.add_component(Capacitor("C1", 150.0, 1.0, capacitance=0.25, voltage=150.0))
    out = lab.simulate_circuit()
    assert out["total_resistor_drop_V"] == pytest.approx(2000.0)
    assert out["stored_energy_J"] == pytest.approx(2812.5)


def test_energy_conservation_full_dump():
    bank = FJHBank(0.25, 150.0, esr_ohm=0.02)
    path = FJHPath(r_sample_ohm=0.20, r_lead_ohm=0.05, r_contact_ohm=0.03)
    split = series_energy_split(bank, path)
    parts = (
        split["energy_in_sample_J"]
        + split["energy_in_leads_J"]
        + split["energy_in_contacts_J"]
        + split["energy_in_esr_J"]
        + split["energy_left_in_bank_J"]
    )
    assert parts == pytest.approx(split["energy_bank_J"], rel=1e-12)
    assert split["energy_left_in_bank_J"] == pytest.approx(0.0, abs=1e-12)
    assert split["energy_bank_J"] == pytest.approx(0.5 * 0.25 * 150.0**2)


def test_partial_dump_leaves_energy_in_bank():
    bank = FJHBank(0.25, 150.0)
    path = FJHPath(0.20, r_lead_ohm=0.05)
    split = series_energy_split(bank, path, t_pulse_s=1e-4)
    assert split["energy_left_in_bank_J"] > 0.1 * split["energy_bank_J"]
    assert split["energy_delivered_J"] + split["energy_left_in_bank_J"] == pytest.approx(
        split["energy_bank_J"]
    )


def test_ohms_law_peak_current():
    bank = FJHBank(0.1, 100.0)
    path = FJHPath(0.4, r_lead_ohm=0.1)
    split = series_energy_split(bank, path)
    assert split["i_peak_A"] == pytest.approx(100.0 / 0.5)
    assert split["tau_s"] == pytest.approx(0.5 * 0.1)


def test_wiring_steals_the_pulse():
    sample = si_c_pellet(0.001)
    C = capacitance_for_target(sample, 150.0, FJHPath(0.20, r_lead_ohm=0.05))
    bank = FJHBank(C, 150.0)
    rows = compare_lead_resistances(bank, 0.20, [0.05, 0.50, 2.00], sample)
    tight, loose, crimp = rows
    assert tight["reached_target"] is True
    assert crimp["reached_target"] is False
    assert tight["T_peak_K"] > loose["T_peak_K"] > crimp["T_peak_K"]
    assert crimp["sample_fraction"] == pytest.approx(0.20 / 2.20, rel=1e-6)
    assert tight["sample_fraction"] == pytest.approx(0.20 / 0.25, rel=1e-6)


def test_missing_property_is_refused():
    with pytest.raises(KeyError):
        electrode("Unobtainium")


def test_electrode_busbar_i2r():
    # 50 cm of 4 mm² copper
    R = electrode_resistance_ohm("Copper", 0.50, 4e-6)
    assert R == pytest.approx(1.68e-8 * 0.50 / 4e-6, rel=1e-9)
    heat = electrode_heating("Copper", 0.50, 4e-6, energy_in_that_resistor_J=200.0)
    assert heat["delta_T_K"] > 0
    assert heat["resistance_ohm"] == pytest.approx(R)


def test_ss304_card_has_fjh_fields():
    card = electrode("SS 304")
    assert card["specific_heat"] == 500.0
    assert card["electrical_resistivity"] == pytest.approx(7.2e-7)


def test_base_materials_db_ss304_no_longer_zero():
    import json
    from pathlib import Path

    raw = json.loads(
        (Path(__file__).resolve().parents[1] / "qulab/labs/engineering/materials_lab/data/materials_db.json").read_text()
    )
    ss = raw["SS 304"]
    assert ss["specific_heat"] == 500.0
    assert ss["electrical_resistivity"] == pytest.approx(7.2e-7)


def test_fire_diagnosis_mentions_leads_when_starved():
    sample = si_c_pellet(0.001)
    result = fire(
        FJHBank(0.05, 150.0),
        FJHPath(0.20, r_lead_ohm=2.0),
        sample,
    )
    blob = " ".join(result.diagnosis).lower()
    assert "lead" in blob or "contact" in blob or "parasitic" in blob
    assert result.reached_target is False


def test_si_c_needed_c_is_hundreds_of_mF():
    sample = si_c_pellet(0.001)
    C = capacitance_for_target(sample, 150.0, FJHPath(0.20, r_lead_ohm=0.05))
    assert 0.15 < C < 0.80


def test_zero_mass_rejected():
    with pytest.raises(ValueError):
        FJHSample(mass_kg=0.0, cp_J_kgK=700.0)


def test_latent_heat_holds_at_melt_until_paid():
    sample = FJHSample(
        mass_kg=0.001,
        cp_J_kgK=1000.0,
        T0_K=300.0,
        T_melt_K=400.0,
        latent_heat_J_kg=1.0e5,
    )
    # Energy to 400 K: 0.001*1000*100 = 100 J. Latent = 100 J.
    from qulab.labs.engineering.fjh_reactor import adiabatic_temperature

    T, melted = adiabatic_temperature(150.0, sample)
    assert melted is False
    assert T == pytest.approx(400.0)
    T2, melted2 = adiabatic_temperature(250.0, sample)
    assert melted2 is True
    assert T2 == pytest.approx(450.0)
