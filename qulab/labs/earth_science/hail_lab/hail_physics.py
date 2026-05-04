import numpy as np
import trimesh
from typing import Dict, Any

class HailPhysicsEngine:
    def __init__(self):
        # Constants for hail properties
        self.ice_density = 917  # kg/m^3
        self.g = 9.81  # m/s^2
        self.rho_air_sea_level = 1.225 # kg/m^3
        self.drag_coeff = 0.45 # Standard for sphere

    def calculate_terminal_velocity(self, diameter_m: float, rho_air: float = 1.225) -> float:
        """
        Calculates terminal velocity using the physics-based formula:
        v_t = sqrt((4 * rho_ice * g * D) / (3 * C_d * rho_air))
        """
        v_t = np.sqrt((4 * self.ice_density * self.g * diameter_m) / (3 * self.drag_coeff * rho_air))
        return float(v_t)

    def calculate_mass_growth_rate(self, diameter_m: float, velocity: float, lwc: float, collection_eff: float = 1.0) -> float:
        """
        Implements mass growth budget concept (Kumjian & Lombardo 2020).
        dm/dt = Area * E * LWC * V_t
        Where Area is cross-sectional area (pi * R^2)
        """
        radius = diameter_m / 2.0
        area = np.pi * (radius ** 2)
        # dm/dt in kg/s
        growth_rate = area * collection_eff * lwc * velocity
        return float(growth_rate)

    def calculate_impact_force(self, mass: float, velocity: float, theta_rad: float, delta_t: float = 0.01) -> float:
        """
        F_impact = (delta_p / delta_t) * cos(theta)
        Where theta is the angle between the hailstone trajectory and the roof normal.
        """
        # delta_p = m * v (assuming full momentum transfer for conservative estimate)
        delta_p = mass * velocity
        # Normal force component: F_impact = (delta_p/delta_t) * cos(theta)
        force = (delta_p / delta_t) * np.cos(theta_rad)
        return float(max(0.0, force))

    def simulate_strike_zone(self, glb_path: str, hail_diameter_m: float = 0.05, num_strikes: int = 100) -> dict:
        """
        Runs a Monte Carlo simulation of hail strikes against a 3D roof mesh.
        """
        try:
            mesh = trimesh.load(glb_path)
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            # Get bounds for random strike positioning
            min_bound, max_bound = mesh.bounds

            strikes = []

            v_term = self.calculate_terminal_velocity(hail_diameter_m)
            mass = (4/3) * np.pi * (hail_diameter_m/2)**3 * self.ice_density

            for _ in range(num_strikes):
                # Random x, y within bounds
                x = np.random.uniform(min_bound[0], max_bound[0])
                y = np.random.uniform(min_bound[1], max_bound[1])
                z_start = max_bound[2] + 1.0 # Start above the roof

                ray_origin = [[x, y, z_start]]
                ray_direction = [[0, 0, -1]] # Falling straight down

                locations, _, index_tri = mesh.ray.intersects_location(
                    ray_origins=ray_origin,
                    ray_directions=ray_direction
                )

                if len(locations) > 0:
                    # Take the first hit point (highest Z)
                    hit_idx = np.argmax(locations[:, 2])
                    hit_point = locations[hit_idx]
                    tri_idx = index_tri[hit_idx]

                    normal = mesh.face_normals[tri_idx]
                    # Angle between vertical (0,0,-1) and normal
                    # cos(theta) = |n . v| / (|n|*|v|)
                    # Since v is (0,0,-1), n . v = -n_z
                    cos_theta = abs(normal[2])
                    theta_rad = np.arccos(np.clip(cos_theta, 0, 1))

                    force = self.calculate_impact_force(mass, v_term, theta_rad)

                    # Threshold for damage (simplified)
                    is_damaged = force > 500 # 500 Newtons as a dummy threshold

                    strikes.append({
                        "point": hit_point.tolist(),
                        "force": float(force),
                        "is_damaged": bool(is_damaged)
                    })

            total_damaged = sum(1 for s in strikes if s["is_damaged"])
            damage_ratio = total_damaged / len(strikes) if strikes else 0

            return {
                "success": True,
                "total_strikes": len(strikes),
                "total_damaged": total_damaged,
                "damage_ratio": damage_ratio,
                "strikes": strikes,
                "hail_diameter_m": hail_diameter_m,
                "terminal_velocity": float(v_term)
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Small self-test
    engine = HailPhysicsEngine()
    v_val = engine.calculate_terminal_velocity(0.05)
    print(f"Terminal velocity for 50mm hail: {v_val:.2f} m/s")

    # Mass growth test
    # Assume LWC = 2.0 g/m^3 = 0.002 kg/m^3
    growth = engine.calculate_mass_growth_rate(0.05, v_val, 0.002)
    print(f"Mass growth rate: {growth:.6f} kg/s")

    # Impact force test
    mass = (4/3) * np.pi * (0.025**3) * 917
    f_val = engine.calculate_impact_force(mass, v_val, np.radians(30))
    print(f"Impact force at 30 deg: {f_val:.2f} N")
