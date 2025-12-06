"""
from nist_constants import *

Units System with Automatic Conversions

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Supports SI, CGS, imperial, and specialized units with automatic conversion.
"""

from typing import Union, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class Unit:
    """Physical unit with conversion to SI base units."""
    name: str
    symbol: str
    to_si: float  # Conversion factor to SI
    dimension: str  # Physical dimension (length, mass, time, etc.)


# Length units
LENGTH_UNITS = {
    'm': Unit('meter', 'm', 1.0, 'length'),
    'cm': Unit('centimeter', 'cm', 0.01, 'length'),
    'mm': Unit('millimeter', 'mm', 0.001, 'length'),
    'um': Unit('micrometer', 'μm', 1e-6, 'length'),
    'nm': Unit('nanometer', 'nm', 1e-9, 'length'),
    'pm': Unit('picometer', 'pm', 1e-12, 'length'),
    'angstrom': Unit('angstrom', 'Å', 1e-10, 'length'),
    'ft': Unit('foot', 'ft', 0.3048, 'length'),
    'in': Unit('inch', 'in', 0.0254, 'length'),
    'mi': Unit('mile', 'mi', 1609.34, 'length'),
    'km': Unit('kilometer', 'km', 1000.0, 'length'),
}

# Mass units
MASS_UNITS = {
    'kg': Unit('kilogram', 'kg', 1.0, 'mass'),
    'g': Unit('gram', 'g', 0.001, 'mass'),
    'mg': Unit('milligram', 'mg', 1e-6, 'mass'),
    'ug': Unit('microgram', 'μg', 1e-9, 'mass'),
    'lb': Unit('pound', 'lb', 0.453592, 'mass'),
    'oz': Unit('ounce', 'oz', 0.0283495, 'mass'),
    'ton': Unit('metric ton', 't', 1000.0, 'mass'),
    'u': Unit('atomic mass unit', 'u', 1.66053906660e-27, 'mass'),
}

# Time units
TIME_UNITS = {
    's': Unit('second', 's', 1.0, 'time'),
    'ms': Unit('millisecond', 'ms', 1e-3, 'time'),
    'us': Unit('microsecond', 'μs', 1e-6, 'time'),
    'ns': Unit('nanosecond', 'ns', 1e-9, 'time'),
    'ps': Unit('picosecond', 'ps', 1e-12, 'time'),
    'fs': Unit('femtosecond', 'fs', 1e-15, 'time'),
    'min': Unit('minute', 'min', 60.0, 'time'),
    'hr': Unit('hour', 'hr', 3600.0, 'time'),
    'day': Unit('day', 'day', 86400.0, 'time'),
}

# Temperature units (special handling for offset units)
TEMPERATURE_UNITS = {
    'K': Unit('kelvin', 'K', 1.0, 'temperature'),
    'C': Unit('celsius', '°C', 1.0, 'temperature'),  # Special conversion
    'F': Unit('fahrenheit', '°F', 5/9, 'temperature'),  # Special conversion
}

# Pressure units
PRESSURE_UNITS = {
    'Pa': Unit('pascal', 'Pa', 1.0, 'pressure'),
    'kPa': Unit('kilopascal', 'kPa', 1000.0, 'pressure'),
    'MPa': Unit('megapascal', 'MPa', 1e6, 'pressure'),
    'GPa': Unit('gigapascal', 'GPa', 1e9, 'pressure'),
    'bar': Unit('bar', 'bar', 100000.0, 'pressure'),
    'atm': Unit('atmosphere', 'atm', 101325.0, 'pressure'),
    'psi': Unit('pounds per square inch', 'psi', 6894.76, 'pressure'),
    'torr': Unit('torr', 'torr', 133.322, 'pressure'),
    'mmHg': Unit('millimeter of mercury', 'mmHg', 133.322, 'pressure'),
}

# Energy units
ENERGY_UNITS = {
    'J': Unit('joule', 'J', 1.0, 'energy'),
    'kJ': Unit('kilojoule', 'kJ', 1000.0, 'energy'),
    'MJ': Unit('megajoule', 'MJ', 1e6, 'energy'),
    'eV': Unit('electronvolt', 'eV', 1.602176634e-19, 'energy'),
    'keV': Unit('kiloelectronvolt', 'keV', 1.602176634e-16, 'energy'),
    'MeV': Unit('megaelectronvolt', 'MeV', 1.602176634e-13, 'energy'),
    'cal': Unit('calorie', 'cal', 4.184, 'energy'),
    'kcal': Unit('kilocalorie', 'kcal', 4184.0, 'energy'),
    'kcal/mol': Unit('kilocalorie per mole', 'kcal/mol', 6.947e-21, 'energy'),
    'kJ/mol': Unit('kilojoule per mole', 'kJ/mol', 1.6605e-21, 'energy'),
    'Ha': Unit('hartree', 'Ha', 4.3597447222071e-18, 'energy'),
}

# Force units
FORCE_UNITS = {
    'N': Unit('newton', 'N', 1.0, 'force'),
    'kN': Unit('kilonewton', 'kN', 1000.0, 'force'),
    'lbf': Unit('pound-force', 'lbf', 4.44822, 'force'),
    'dyn': Unit('dyne', 'dyn', 1e-5, 'force'),
}

# Power units
POWER_UNITS = {
    'W': Unit('watt', 'W', 1.0, 'power'),
    'kW': Unit('kilowatt', 'kW', 1000.0, 'power'),
    'MW': Unit('megawatt', 'MW', 1e6, 'power'),
    'hp': Unit('horsepower', 'hp', 745.7, 'power'),
}

# Velocity units
VELOCITY_UNITS = {
    'm/s': Unit('meters per second', 'm/s', 1.0, 'velocity'),
    'km/h': Unit('kilometers per hour', 'km/h', 1/3.6, 'velocity'),
    'mph': Unit('miles per hour', 'mph', 0.44704, 'velocity'),
    'ft/s': Unit('feet per second', 'ft/s', 0.3048, 'velocity'),
}

ALL_UNITS = {
    **LENGTH_UNITS,
    **MASS_UNITS,
    **TIME_UNITS,
    **TEMPERATURE_UNITS,
    **PRESSURE_UNITS,
    **ENERGY_UNITS,
    **FORCE_UNITS,
    **POWER_UNITS,
    **VELOCITY_UNITS,
}


def convert(value: Union[float, np.ndarray],
           from_unit: str,
           to_unit: str) -> Union[float, np.ndarray]:
    """
    Convert value from one unit to another.

    Args:
        value: Numerical value or array to convert
        from_unit: Source unit symbol
        to_unit: Target unit symbol

    Returns:
        Converted value

    Raises:
        ValueError: If units are incompatible or unknown

    Examples:
        >>> convert(1.0, 'm', 'cm')
        100.0
        >>> convert(100.0, 'C', 'K')
        373.15
        >>> convert(1.0, 'eV', 'J')
        1.602176634e-19
    """
    if from_unit == to_unit:
        return value

    # Special handling for temperature (offset units)
    if from_unit in TEMPERATURE_UNITS and to_unit in TEMPERATURE_UNITS:
        return _convert_temperature(value, from_unit, to_unit)

    if from_unit not in ALL_UNITS:
        raise ValueError(f"Unknown unit: {from_unit}")
    if to_unit not in ALL_UNITS:
        raise ValueError(f"Unknown unit: {to_unit}")

    from_unit_obj = ALL_UNITS[from_unit]
    to_unit_obj = ALL_UNITS[to_unit]

    if from_unit_obj.dimension != to_unit_obj.dimension:
        raise ValueError(
            f"Incompatible units: {from_unit} ({from_unit_obj.dimension}) "
            f"and {to_unit} ({to_unit_obj.dimension})"
        )

    # Convert: from_unit → SI → to_unit
    si_value = value * from_unit_obj.to_si
    result = si_value / to_unit_obj.to_si

    return result


def _convert_temperature(value: Union[float, np.ndarray],
                        from_unit: str,
                        to_unit: str) -> Union[float, np.ndarray]:
    """Convert temperature with offset handling."""
    # First convert to Kelvin
    if from_unit == 'K':
        kelvin = value
    elif from_unit == 'C':
        kelvin = value + 273.15
    elif from_unit == 'F':
        kelvin = (value - 32) * 5/9 + 273.15
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")

    # Then convert from Kelvin to target
    if to_unit == 'K':
        return kelvin
    elif to_unit == 'C':
        return kelvin - 273.15
    elif to_unit == 'F':
        return (kelvin - 273.15) * 9/5 + 32
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")


class Quantity:
    """Physical quantity with value and units."""

    def __init__(self, value: Union[float, np.ndarray], units: str):
        """
        Create a physical quantity.

        Args:
            value: Numerical value
            units: Unit symbol
        """
        self.value = value
        self.units = units

        if units not in ALL_UNITS and units not in TEMPERATURE_UNITS:
            raise ValueError(f"Unknown unit: {units}")

    def to(self, target_units: str) -> 'Quantity':
        """Convert to different units."""
        converted_value = convert(self.value, self.units, target_units)
        return Quantity(converted_value, target_units)

    def __repr__(self) -> str:
        return f"{self.value} {self.units}"

    def __add__(self, other: 'Quantity') -> 'Quantity':
        """Add two quantities (converts to same units)."""
        other_converted = other.to(self.units)
        return Quantity(self.value + other_converted.value, self.units)

    def __sub__(self, other: 'Quantity') -> 'Quantity':
        """Subtract two quantities (converts to same units)."""
        other_converted = other.to(self.units)
        return Quantity(self.value - other_converted.value, self.units)

    def __mul__(self, scalar: float) -> 'Quantity':
        """Multiply by scalar."""
        return Quantity(self.value * scalar, self.units)

    def __rmul__(self, scalar: float) -> 'Quantity':
        """Right multiply by scalar."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> 'Quantity':
        """Divide by scalar."""
        return Quantity(self.value / scalar, self.units)


if __name__ == "__main__":
    print("QuLab Infinite Units System")
    print("=" * 80)

    # Length conversions
    print("\nLength conversions:")
    print(f"1 m = {convert(1, 'm', 'cm')} cm")
    print(f"1 m = {convert(1, 'm', 'mm')} mm")
    print(f"1 m = {convert(1, 'm', 'nm'):.2e} nm")
    print(f"1 m = {convert(1, 'm', 'angstrom'):.2e} Å")
    print(f"1 m = {convert(1, 'm', 'ft'):.4f} ft")
    print(f"1 m = {convert(1, 'm', 'in'):.4f} in")

    # Temperature conversions
    print("\nTemperature conversions:")
    print(f"0 °C = {convert(0, 'C', 'K')} K")
    print(f"0 °C = {convert(0, 'C', 'F')} °F")
    print(f"100 °C = {convert(100, 'C', 'K')} K")
    print(f"100 °C = {convert(100, 'C', 'F')} °F")
    print(f"-200 °C = {convert(-200, 'C', 'K')} K")

    # Pressure conversions
    print("\nPressure conversions:")
    print(f"1 atm = {convert(1, 'atm', 'Pa')} Pa")
    print(f"1 atm = {convert(1, 'atm', 'bar')} bar")
    print(f"1 atm = {convert(1, 'atm', 'psi'):.2f} psi")
    print(f"1 atm = {convert(1, 'atm', 'torr'):.2f} torr")

    # Energy conversions
    print("\nEnergy conversions:")
    print(f"1 eV = {convert(1, 'eV', 'J'):.4e} J")
    print(f"1 kcal/mol = {convert(1, 'kcal/mol', 'kJ/mol'):.4f} kJ/mol")
    print(f"1 Ha = {convert(1, 'Ha', 'eV'):.4f} eV")

    # Velocity conversions
    print("\nVelocity conversions:")
    print(f"30 mph = {convert(30, 'mph', 'm/s'):.2f} m/s")
    print(f"100 km/h = {convert(100, 'km/h', 'm/s'):.2f} m/s")

    # Quantity examples
    print("\nQuantity arithmetic:")
    q1 = Quantity(100, 'cm')
    q2 = Quantity(1, 'm')
    print(f"{q1} + {q2} = {q1 + q2}")
    print(f"{q1} to meters = {q1.to('m')}")
    print(f"Temperature: {Quantity(-200, 'C').to('K')}")
