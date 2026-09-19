"""
Electrical helpers used by the FJH reactor path.

What this will do
- Ohm's law: V = I R, R = V / I, P = I² R
- Capacitor bank energy: E = ½ C V²
- Series resistance and voltage drops that actually add up

What this will not do
- SPICE-quality transient simulation
- Invent an inductance on a capacitor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CircuitElement:
    name: str
    voltage_rating: float
    current_rating: float


@dataclass
class Resistor(CircuitElement):
    resistance: float
    power_rating: float = 2.5


@dataclass
class Capacitor(CircuitElement):
    capacitance: float
    voltage: float | None = None


@dataclass
class Inductor(CircuitElement):
    inductance: float


@dataclass
class VoltageSource:
    voltage: float
    internal_resistance: float = 0.0
    current: float | None = None


@dataclass
class CurrentSource:
    current: float
    internal_resistance: float = 0.0


@dataclass
class Battery(VoltageSource):
    capacity_Ah: float = 0.0


def calculate_resistance(voltage: float, current: float) -> float:
    if current == 0:
        raise ValueError("current must be nonzero to compute resistance")
    return voltage / current


def voltage_from_ir(resistance: float, current: float) -> float:
    return resistance * current


def calculate_power(resistance: float, current: float) -> float:
    return (current**2) * resistance


def capacitor_energy(capacitance: float, voltage: float) -> float:
    if capacitance < 0:
        raise ValueError("capacitance cannot be negative")
    return 0.5 * capacitance * voltage**2


@dataclass
class ElectricalLab:
    components: list[Any] = field(default_factory=list)

    def add_component(self, component: Any) -> None:
        self.components.append(component)

    def simulate_circuit(self) -> dict[str, Any]:
        """
        Series-ish bookkeeping. Resistors drop I*R. Capacitors store ½CV².
        Returns a dict; also prints a one-line summary.
        """
        drops: list[dict[str, Any]] = []
        total_drop = 0.0
        total_power = 0.0
        stored_energy = 0.0

        for component in self.components:
            if isinstance(component, Resistor):
                v = voltage_from_ir(component.resistance, component.current_rating)
                p = calculate_power(component.resistance, component.current_rating)
                total_drop += v
                total_power += p
                drops.append(
                    {
                        "name": component.name,
                        "type": "resistor",
                        "voltage_drop_V": v,
                        "power_W": p,
                    }
                )
            elif isinstance(component, Capacitor):
                voltage = component.voltage if component.voltage is not None else component.voltage_rating
                energy = capacitor_energy(component.capacitance, voltage)
                stored_energy += energy
                drops.append(
                    {
                        "name": component.name,
                        "type": "capacitor",
                        "energy_J": energy,
                        "voltage_V": voltage,
                    }
                )
            elif isinstance(component, Inductor):
                energy = 0.5 * component.inductance * component.current_rating**2
                stored_energy += energy
                drops.append(
                    {
                        "name": component.name,
                        "type": "inductor",
                        "energy_J": energy,
                    }
                )
            elif isinstance(component, VoltageSource):
                if component.current is not None:
                    v_int = voltage_from_ir(component.internal_resistance, component.current)
                else:
                    v_int = 0.0
                drops.append(
                    {
                        "name": "source",
                        "type": "voltage_source",
                        "voltage_V": component.voltage,
                        "internal_drop_V": v_int,
                    }
                )

        summary = {
            "total_resistor_drop_V": total_drop,
            "total_resistor_power_W": total_power,
            "stored_energy_J": stored_energy,
            "elements": drops,
        }
        print(
            f"Total resistor drop: {total_drop:.4f} V; "
            f"power: {total_power:.4f} W; stored energy: {stored_energy:.6f} J"
        )
        return summary


def run_demo() -> dict[str, Any]:
    lab = ElectricalLab()
    lab.add_component(Resistor("R1", 5.0, 2.0, resistance=1000, power_rating=5))
    lab.add_component(Capacitor("C1", 150.0, 1.0, capacitance=0.25, voltage=150.0))
    lab.add_component(Inductor("L1", 5.0, 2.0, inductance=1e-3))
    lab.add_component(VoltageSource(12.0, internal_resistance=0.01, current=2.0))
    return lab.simulate_circuit()


if __name__ == "__main__":
    run_demo()
