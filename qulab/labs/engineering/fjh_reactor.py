"""
Flash Joule Heating reactor model.

This exists because the useful part of QuLabInfinite, for this hardware, is
diagnosing whether a capacitor pulse heated the pellet or the crimps.

What this will tell you
- Bank energy E = 1/2 C V^2
- Series partition among sample, leads, contacts, and capacitor ESR
- Peak current, RC time constant, energy delivered before a switch opens
- Adiabatic (and optional radiative) sample temperature
- Electrode I^2R if you give bus geometry
- Plain-language wiring diagnosis

What this will not tell you
- Whether you made β-SiC nanowires
- Microstructure, phase purity, or yield
- A 150 V / 500 ms label by itself. Pulse width is not energy. You must
  give C (and mass).

No empirical latent-heat fudge factors. If a property is missing, it errors
instead of silently using 0.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

SIGMA = 5.670374419e-8  # Stefan-Boltzmann, W/(m^2 K^4)
ELECTRODE_PATH = Path(__file__).resolve().parent / "data" / "fjh_electrodes.json"


def _load_electrodes() -> dict[str, dict[str, Any]]:
    raw = json.loads(ELECTRODE_PATH.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


ELECTRODES = _load_electrodes()


def require_positive(name: str, value: float) -> float:
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return float(value)


def electrode(name: str) -> dict[str, Any]:
    if name not in ELECTRODES:
        known = ", ".join(sorted(ELECTRODES))
        raise KeyError(f"Unknown FJH material {name!r}. Known: {known}")
    card = ELECTRODES[name]
    for key in ("specific_heat", "electrical_resistivity", "density"):
        val = card.get(key, 0) or 0
        if val <= 0:
            raise ValueError(
                f"{name}.{key} is {val}. Refusing to compute with a missing property."
            )
    return card


@dataclass
class FJHBank:
    """Capacitor bank. Energy is 1/2 C V0^2. Pulse width is not energy."""

    capacitance_F: float
    v0_V: float
    esr_ohm: float = 0.0

    def __post_init__(self) -> None:
        require_positive("capacitance_F", self.capacitance_F)
        require_positive("v0_V", self.v0_V)
        if self.esr_ohm < 0:
            raise ValueError("esr_ohm cannot be negative")

    @property
    def energy_J(self) -> float:
        return 0.5 * self.capacitance_F * self.v0_V**2


@dataclass
class FJHPath:
    """Series resistances around the sample. Sample R can be a number or R(T)."""

    r_sample_ohm: float | Callable[[float], float]
    r_lead_ohm: float = 0.0
    r_contact_ohm: float = 0.0

    def parasitic(self) -> float:
        if self.r_lead_ohm < 0 or self.r_contact_ohm < 0:
            raise ValueError("lead/contact resistance cannot be negative")
        return float(self.r_lead_ohm + self.r_contact_ohm)

    def sample_at(self, T: float) -> float:
        r = self.r_sample_ohm(T) if callable(self.r_sample_ohm) else self.r_sample_ohm
        return require_positive("r_sample_ohm", r)


@dataclass
class FJHSample:
    mass_kg: float
    cp_J_kgK: float
    T0_K: float = 298.15
    area_m2: float = 0.0
    emissivity: float = 0.0
    T_melt_K: float | None = None
    latent_heat_J_kg: float | None = None
    T_target_K: float | None = None

    def __post_init__(self) -> None:
        require_positive("mass_kg", self.mass_kg)
        require_positive("cp_J_kgK", self.cp_J_kgK)
        require_positive("T0_K", self.T0_K)
        if self.area_m2 < 0 or self.emissivity < 0:
            raise ValueError("area/emissivity cannot be negative")


@dataclass
class FJHResult:
    energy_bank_J: float
    energy_delivered_J: float
    energy_in_sample_J: float
    energy_in_leads_J: float
    energy_in_contacts_J: float
    energy_in_esr_J: float
    energy_left_in_bank_J: float
    sample_fraction: float
    parasitic_fraction: float
    r_sample_ohm: float
    r_total_ohm: float
    tau_s: float
    i_peak_A: float
    t_pulse_s: float | None
    T_adiabatic_K: float
    T_peak_K: float
    melted: bool
    reached_target: bool | None
    diagnosis: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def series_energy_split(
    bank: FJHBank,
    path: FJHPath,
    t_pulse_s: float | None = None,
    T: float = 298.15,
) -> dict[str, float]:
    """
    Constant-R series RC dump.

    V(t) = V0 exp(-t/RC)
    E_left(t) = E0 exp(-2t/RC)
    E_delivered = E0 - E_left
    Each resistor takes energy in proportion to its R.
    """
    r_s = path.sample_at(T)
    r_lead = float(path.r_lead_ohm)
    r_contact = float(path.r_contact_ohm)
    r_esr = float(bank.esr_ohm)
    r_total = r_s + r_lead + r_contact + r_esr
    require_positive("r_total", r_total)

    tau = r_total * bank.capacitance_F
    i_peak = bank.v0_V / r_total
    e0 = bank.energy_J

    if t_pulse_s is None:
        delivered_frac = 1.0
        t_used = None
    else:
        require_positive("t_pulse_s", t_pulse_s)
        delivered_frac = 1.0 - math.exp(-2.0 * t_pulse_s / tau)
        t_used = t_pulse_s

    e_del = e0 * delivered_frac
    return {
        "r_sample": r_s,
        "r_lead": r_lead,
        "r_contact": r_contact,
        "r_esr": r_esr,
        "r_total": r_total,
        "tau_s": tau,
        "i_peak_A": i_peak,
        "energy_bank_J": e0,
        "energy_delivered_J": e_del,
        "energy_left_in_bank_J": e0 - e_del,
        "energy_in_sample_J": e_del * (r_s / r_total),
        "energy_in_leads_J": e_del * (r_lead / r_total),
        "energy_in_contacts_J": e_del * (r_contact / r_total),
        "energy_in_esr_J": e_del * (r_esr / r_total),
        "sample_fraction": r_s / r_total,
        "parasitic_fraction": (r_lead + r_contact + r_esr) / r_total,
        "t_pulse_s": t_used if t_used is not None else float("inf"),
    }


def adiabatic_temperature(energy_J: float, sample: FJHSample) -> tuple[float, bool]:
    """
    Convert sample energy to temperature, consuming real latent heat at T_melt.

    Returns (T, melted).
    """
    heat_capacity = sample.mass_kg * sample.cp_J_kgK
    T = sample.T0_K
    remaining = energy_J
    melted = False

    if sample.T_melt_K and sample.latent_heat_J_kg:
        require_positive("T_melt_K", sample.T_melt_K)
        require_positive("latent_heat_J_kg", sample.latent_heat_J_kg)
        to_melt = heat_capacity * max(sample.T_melt_K - T, 0.0)
        if remaining <= to_melt:
            return T + remaining / heat_capacity, False
        remaining -= to_melt
        T = sample.T_melt_K
        latent = sample.mass_kg * sample.latent_heat_J_kg
        if remaining < latent:
            return T, False
        remaining -= latent
        melted = True

    return T + remaining / heat_capacity, melted


def radiative_peak_temperature(
    bank: FJHBank,
    path: FJHPath,
    sample: FJHSample,
    t_pulse_s: float | None = None,
    dt: float = 2e-5,
) -> tuple[float, bool]:
    """
    Forward-Euler dump with radiation. Constant R evaluated at T0.
    Used as a sanity check that radiation cannot save a starved pellet
    on a ~ms pulse, and can matter on a long tail.
    """
    split = series_energy_split(bank, path, t_pulse_s=None, T=sample.T0_K)
    r_s = split["r_sample"]
    r_total = split["r_total"]
    tau = split["tau_s"]
    i_peak = split["i_peak_A"]
    horizon = t_pulse_s if t_pulse_s is not None else 8.0 * tau
    n = max(int(horizon / dt), 50)
    dt = horizon / n

    T = sample.T0_K
    latent_buf = 0.0
    melted = False
    heat_capacity = sample.mass_kg * sample.cp_J_kgK
    t = 0.0
    for _ in range(n):
        i = i_peak * math.exp(-t / tau)
        p_joule = i * i * r_s
        p_rad = 0.0
        if sample.area_m2 > 0 and sample.emissivity > 0:
            p_rad = sample.emissivity * SIGMA * sample.area_m2 * (T**4 - sample.T0_K**4)
        p_net = p_joule - p_rad
        dU = p_net * dt

        if (
            sample.T_melt_K
            and sample.latent_heat_J_kg
            and T >= sample.T_melt_K
            and not melted
        ):
            latent_need = sample.mass_kg * sample.latent_heat_J_kg
            latent_buf += max(dU, 0.0)
            T = sample.T_melt_K
            if latent_buf >= latent_need:
                melted = True
                leftover = latent_buf - latent_need
                T += leftover / heat_capacity
        else:
            T += dU / heat_capacity
            if T < 1.0:
                T = 1.0
        t += dt
        if t_pulse_s is not None and t >= t_pulse_s:
            break
    return T, melted


def diagnose(split: dict[str, float], sample: FJHSample, T_peak: float) -> list[str]:
    notes: list[str] = []
    frac = split["sample_fraction"]
    tau = split["tau_s"]
    t_pulse = split["t_pulse_s"]
    i_peak = split["i_peak_A"]

    if frac < 0.5:
        notes.append(
            f"Most of the pulse is not in the pellet ({frac*100:.0f}% sample / "
            f"{split['parasitic_fraction']*100:.0f}% leads+contacts+ESR). "
            "Tighten contacts, shorten leads, or raise sample R relative to the bus."
        )
    elif frac < 0.75:
        notes.append(
            f"Only {frac*100:.0f}% of delivered energy is in the sample. "
            "Worth chasing contact resistance before blaming the precursor."
        )
    else:
        notes.append(
            f"{frac*100:.0f}% of delivered energy is in the sample. "
            "If the shot still looks cold, look at C, V, mass, or switch timing."
        )

    if math.isfinite(t_pulse) and t_pulse < 3.0 * tau:
        dumped = split["energy_delivered_J"] / split["energy_bank_J"]
        notes.append(
            f"Switch window {t_pulse*1e3:.1f} ms is short vs RC τ={tau*1e3:.1f} ms "
            f"({dumped*100:.0f}% of the bank dumped). Pulse width is cutting the energy."
        )
    elif math.isfinite(t_pulse) and t_pulse > 8.0 * tau:
        notes.append(
            f"RC τ={tau*1e3:.1f} ms << listed pulse {t_pulse*1e3:.1f} ms. "
            "Energy is set by ½CV², not by the 500 ms label."
        )

    if sample.T_target_K is not None:
        if T_peak >= sample.T_target_K:
            notes.append(
                f"Peak T {T_peak:.0f} K meets target {sample.T_target_K:.0f} K "
                "(adiabatic/radiative estimate, not a pyrometer)."
            )
        else:
            notes.append(
                f"Peak T {T_peak:.0f} K misses target {sample.T_target_K:.0f} K. "
                "Raise C or V, cut parasitic R, or use less mass."
            )

    if i_peak > 2000:
        notes.append(
            f"Peak current {i_peak:.0f} A. Check switch, cable ampacity, and "
            "whether the leads will vaporize before the pellet heats."
        )
    return notes


def fire(
    bank: FJHBank,
    path: FJHPath,
    sample: FJHSample,
    t_pulse_s: float | None = None,
    include_radiation: bool = False,
) -> FJHResult:
    split = series_energy_split(bank, path, t_pulse_s=t_pulse_s, T=sample.T0_K)
    T_ad, melted_ad = adiabatic_temperature(split["energy_in_sample_J"], sample)
    if include_radiation and sample.area_m2 > 0 and sample.emissivity > 0:
        T_peak, melted = radiative_peak_temperature(bank, path, sample, t_pulse_s)
    else:
        T_peak, melted = T_ad, melted_ad

    reached = None
    if sample.T_target_K is not None:
        reached = T_peak >= sample.T_target_K

    notes = [
        "This is an energy-partition model, not a materials-discovery engine.",
        "R_sample is whatever you measured (or guessed) for the pellet + paper + contacts.",
    ]
    return FJHResult(
        energy_bank_J=split["energy_bank_J"],
        energy_delivered_J=split["energy_delivered_J"],
        energy_in_sample_J=split["energy_in_sample_J"],
        energy_in_leads_J=split["energy_in_leads_J"],
        energy_in_contacts_J=split["energy_in_contacts_J"],
        energy_in_esr_J=split["energy_in_esr_J"],
        energy_left_in_bank_J=split["energy_left_in_bank_J"],
        sample_fraction=split["sample_fraction"],
        parasitic_fraction=split["parasitic_fraction"],
        r_sample_ohm=split["r_sample"],
        r_total_ohm=split["r_total"],
        tau_s=split["tau_s"],
        i_peak_A=split["i_peak_A"],
        t_pulse_s=t_pulse_s,
        T_adiabatic_K=T_ad,
        T_peak_K=T_peak,
        melted=melted,
        reached_target=reached,
        diagnosis=diagnose(split, sample, T_peak),
        notes=notes,
    )


def electrode_resistance_ohm(material: str, length_m: float, area_m2: float) -> float:
    card = electrode(material)
    require_positive("length_m", length_m)
    require_positive("area_m2", area_m2)
    return card["electrical_resistivity"] * length_m / area_m2


def electrode_heating(
    material: str,
    length_m: float,
    area_m2: float,
    energy_in_that_resistor_J: float,
) -> dict[str, float]:
    card = electrode(material)
    require_positive("length_m", length_m)
    require_positive("area_m2", area_m2)
    mass = card["density"] * area_m2 * length_m
    R = electrode_resistance_ohm(material, length_m, area_m2)
    dT = energy_in_that_resistor_J / (mass * card["specific_heat"])
    return {
        "resistance_ohm": R,
        "mass_kg": mass,
        "delta_T_K": dT,
        "T_if_from_298K": 298.15 + dT,
    }


def si_c_pellet(mass_kg: float = 0.001) -> FJHSample:
    """
    1:1 molar Si:C mix. Mass-weighted cp. Si latent heat only.
    This is a thermal-mass estimate, not a reaction model.
    """
    require_positive("mass_kg", mass_kg)
    m_si_frac = 28.085 / (28.085 + 12.011)
    m_c_frac = 1.0 - m_si_frac
    si = electrode("Silicon")
    carbon = electrode("Carbon")
    cp = m_si_frac * si["specific_heat"] + m_c_frac * carbon["specific_heat"]
    return FJHSample(
        mass_kg=mass_kg,
        cp_J_kgK=cp,
        T0_K=298.15,
        T_melt_K=si["melting_point"],
        latent_heat_J_kg=m_si_frac * si["latent_heat_fusion"],
        T_target_K=2500.0,
    )


def capacitance_for_target(
    sample: FJHSample,
    v0_V: float,
    path: FJHPath,
    T_target_K: float | None = None,
    esr_ohm: float = 0.0,
) -> float:
    """C required so a full dump reaches T_target, given the series split."""
    target = T_target_K or sample.T_target_K
    if target is None:
        raise ValueError("T_target_K is required")
    require_positive("v0_V", v0_V)
    dummy = FJHBank(capacitance_F=1.0, v0_V=v0_V, esr_ohm=esr_ohm)
    split = series_energy_split(dummy, path, t_pulse_s=None, T=sample.T0_K)
    heat_capacity = sample.mass_kg * sample.cp_J_kgK
    e_need = heat_capacity * (target - sample.T0_K)
    if sample.T_melt_K and sample.latent_heat_J_kg and target > sample.T_melt_K:
        e_need += sample.mass_kg * sample.latent_heat_J_kg
    if split["sample_fraction"] <= 0:
        raise ValueError("sample fraction is zero")
    e_bank = e_need / split["sample_fraction"]
    return 2.0 * e_bank / v0_V**2


def compare_lead_resistances(
    bank: FJHBank,
    r_sample_ohm: float,
    lead_ohms: list[float],
    sample: FJHSample,
    t_pulse_s: float | None = None,
    r_contact_ohm: float = 0.0,
) -> list[dict[str, Any]]:
    rows = []
    for r_lead in lead_ohms:
        result = fire(
            bank,
            FJHPath(r_sample_ohm=r_sample_ohm, r_lead_ohm=r_lead, r_contact_ohm=r_contact_ohm),
            sample,
            t_pulse_s=t_pulse_s,
        )
        rows.append(
            {
                "r_lead_ohm": r_lead,
                "sample_fraction": result.sample_fraction,
                "T_peak_K": result.T_peak_K,
                "i_peak_A": result.i_peak_A,
                "reached_target": result.reached_target,
                "diagnosis": result.diagnosis[0] if result.diagnosis else "",
            }
        )
    return rows


def _print_result(result: FJHResult) -> None:
    print(f"Bank energy:     {result.energy_bank_J:10.1f} J")
    print(f"Delivered:       {result.energy_delivered_J:10.1f} J")
    print(f"In sample:       {result.energy_in_sample_J:10.1f} J  ({result.sample_fraction*100:.1f}%)")
    print(f"In leads:        {result.energy_in_leads_J:10.1f} J")
    print(f"In contacts:     {result.energy_in_contacts_J:10.1f} J")
    print(f"In ESR:          {result.energy_in_esr_J:10.1f} J")
    print(f"Left in bank:    {result.energy_left_in_bank_J:10.1f} J")
    print(f"R_sample/total:  {result.r_sample_ohm:.4f} / {result.r_total_ohm:.4f} ohm")
    print(f"Peak current:    {result.i_peak_A:.1f} A")
    print(f"RC tau:          {result.tau_s*1e3:.2f} ms")
    print(f"T adiabatic:     {result.T_adiabatic_K:.0f} K")
    print(f"T peak:          {result.T_peak_K:.0f} K")
    print(f"Melted:          {result.melted}")
    print(f"Reached target:  {result.reached_target}")
    print("Diagnosis:")
    for line in result.diagnosis:
        print(f"  - {line}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="FJH energy-partition and wiring diagnosis. Pulse width is not energy."
    )
    p.add_argument("--C", dest="C", type=float, required=False, help="Capacitance in farads")
    p.add_argument("--V", dest="V", type=float, default=150.0, help="Charge voltage")
    p.add_argument("--mass", type=float, default=0.001, help="Pellet mass in kg")
    p.add_argument("--R-sample", dest="R_sample", type=float, default=0.20)
    p.add_argument("--R-lead", dest="R_lead", type=float, default=0.05)
    p.add_argument("--R-contact", dest="R_contact", type=float, default=0.0)
    p.add_argument("--ESR", dest="esr", type=float, default=0.0)
    p.add_argument("--t-pulse", dest="t_pulse", type=float, default=None, help="Switch window in seconds")
    p.add_argument("--T-target", dest="T_target", type=float, default=2500.0)
    p.add_argument("--compare-leads", action="store_true", help="Sweep 50 mΩ / 500 mΩ / 2 Ω leads")
    p.add_argument("--needed-C", action="store_true", help="Print C that reaches T-target at this V")
    args = p.parse_args(argv)

    sample = si_c_pellet(args.mass)
    sample.T_target_K = args.T_target
    path = FJHPath(r_sample_ohm=args.R_sample, r_lead_ohm=args.R_lead, r_contact_ohm=args.R_contact)

    if args.needed_C:
        C = capacitance_for_target(sample, args.V, path, esr_ohm=args.esr)
        print(f"C needed for {args.T_target:.0f} K at {args.V:.1f} V: {C*1e3:.1f} mF")
        print("(Full dump assumed. Add margin for switch cut-off and radiation.)")
        return 0

    if args.compare_leads:
        if args.C is None:
            args.C = capacitance_for_target(sample, args.V, FJHPath(args.R_sample, 0.05), esr_ohm=args.esr)
            print(f"No --C given; using C={args.C*1e3:.1f} mF so 50 mΩ leads can hit target.\n")
        bank = FJHBank(args.C, args.V, args.esr)
        rows = compare_lead_resistances(
            bank, args.R_sample, [0.05, 0.50, 2.00], sample, args.t_pulse, args.R_contact
        )
        print(f"{'R_lead':>10} {'% sample':>10} {'T_peak K':>10} {'I_peak A':>10}  reached")
        for row in rows:
            print(
                f"{row['r_lead_ohm']:10.3f} {row['sample_fraction']*100:9.1f}% "
                f"{row['T_peak_K']:10.0f} {row['i_peak_A']:10.1f}  {row['reached_target']}"
            )
        return 0

    if args.C is None:
        p.error("C is required unless you pass --compare-leads or --needed-C")

    result = fire(FJHBank(args.C, args.V, args.esr), path, sample, t_pulse_s=args.t_pulse)
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
