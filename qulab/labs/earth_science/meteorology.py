"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

METEOROLOGY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from scipy.constants import pi, R, g, k


@dataclass
class WindProfile:
    u: np.ndarray  # zonal wind speeds (m/s)
    v: np.ndarray  # meridional wind speeds (m/s)
    z: np.ndarray  # altitude array (km)

    def __post_init__(self):
        assert self.u.shape == self.v.shape == self.z.shape, "Shape mismatch between u, v and z arrays."
        assert len(self.u) > 1, "At least two levels required for a wind profile."

@dataclass
class AirParcel:
    T: float = 298.15  # Temperature (K)
    p: float = 101325.  # Pressure (Pa)
    qv: float = 1e-4   # Specific humidity (kg/kg)
    z: float = 0       # Altitude (m)

    def ascent(self, dz=1., lapse_rate=-0.0065):
        self.T += lapse_rate * dz
        self.p *= np.exp(-g * (self.z + dz) / R / self.T)
        self.qv /= self.p  # Assuming qv scales with pressure

    def __str__(self):
        return f"T: {self.T:.2f}K, p: {self.p/100:.2f}hPa, RH: {self.relative_humidity():.2%}, z: {self.z/1e3:.2f}km"

    def relative_humidity(self):
        e = 6.11 * np.exp(5417.7530 * (1/273.15 - 1/self.T))
        rh = self.qv / (k * self.T) * e
        return rh

@dataclass
class StabilityIndex:
    u: np.ndarray
    v: np.ndarray
    z: np.ndarray
    T_profile: np.ndarray
    p_profile: np.ndarray

    def brunt_vaisala_freq(self, dz=10):
        N = np.sqrt(-g / self.T_profile[0] * (self.T_profile[:-dz] - self.T_profile[dz:]) / dz)
        return N

def create_stability_index(wind_prof: WindProfile, T_p: np.ndarray, p_p: np.ndarray) -> StabilityIndex:
    assert wind_prof.u.shape == T_p.shape and wind_prof.v.shape == p_p.shape
    return StabilityIndex(u=wind_prof.u, v=wind_prof.v, z=wind_prof.z * 1000., T_profile=T_p, p_profile=p_p)

def simulate_atmospheric_boundary_layer(height: int = 3000):
    dz = 100  # m
    z_levels = np.arange(0, height + dz, dz)
    u_zonal = 5 * (z_levels / height) ** 2  # Example profile function
    v_meridional = -2 * (z_levels / height) ** 2  # Example profile function
    T_atm = np.ones_like(z_levels) * 298.15  # Assume a constant temperature for simplicity

    # Create a WindProfile instance
    wind_profile_example = WindProfile(u=np.array(u_zonal, dtype=np.float64), v=np.array(v_meridional, dtype=np.float64),
                                       z=np.array(z_levels / 1000., dtype=np.float64))

    # Create an air parcel and ascend it through the boundary layer
    parcel_example = AirParcel(p=95000)
    print("Initial Conditions:")
    print(parcel_example)

    for dz in range(0, height + dz, int(dz / 2)):
        parcel_example.ascent(dz=dz / 1000)
        if (dz % 100 == 0):
            print(f"At z={parcel_example.z/1e3:.2f}km: {parcel_example}")

def run_demo():
    simulate_atmospheric_boundary_layer()

if __name__ == '__main__':
    run_demo()