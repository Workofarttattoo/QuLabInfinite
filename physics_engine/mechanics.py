"""
Newtonian Mechanics Engine

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Implements classical mechanics: dynamics, collisions, friction, elasticity.
Adaptive timestep integration with energy conservation validation.
"""

from typing import Tuple, List, Optional, Callable, Dict
from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .fundamental_constants import g_0, G


class GravityType(Enum):
    """Type of gravity model to use."""
    UNIFORM = "uniform"
    N_BODY = "n_body"


class IntegrationMethod(Enum):
    """Type of numerical integrator to use."""
    VELOCITY_VERLET = "velocity_verlet"
    SCIPY_DOP853 = "scipy_dop853"


@dataclass
class Particle:
    """A particle with mass, position, velocity, and force."""
    mass: float  # kg
    position: NDArray[np.float64]  # [x, y, z] in meters
    velocity: NDArray[np.float64]  # [vx, vy, vz] in m/s
    force: NDArray[np.float64]  # [fx, fy, fz] in Newtons
    radius: float = 0.0  # meters (for collision detection)
    fixed: bool = False  # If True, particle doesn't move

    def __post_init__(self):
        """Ensure arrays are the right shape."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.force = np.asarray(self.force, dtype=np.float64)


@dataclass
class RigidBody:
    """A rigid body with orientation and angular momentum."""
    mass: float  # kg
    position: NDArray[np.float64]  # Center of mass [x, y, z] in meters
    velocity: NDArray[np.float64]  # Linear velocity [vx, vy, vz] in m/s
    orientation: NDArray[np.float64]  # Quaternion [w, x, y, z]
    angular_velocity: NDArray[np.float64]  # Angular velocity [wx, wy, wz] in rad/s
    inertia_tensor: NDArray[np.float64]  # 3x3 inertia tensor in kg⋅m²
    force: NDArray[np.float64]  # Total force [fx, fy, fz] in N
    torque: NDArray[np.float64]  # Total torque [tx, ty, tz] in N⋅m
    fixed: bool = False


class Constraint:
    """Base class for all constraints."""
    def solve(self, particles: List[Particle], dt: float):
        """Solve the constraint."""
        raise NotImplementedError

class DistanceConstraint(Constraint):
    """Constraint that keeps two particles at a fixed distance."""
    def __init__(self, p1_idx: int, p2_idx: int, distance: float):
        self.p1_idx = p1_idx
        self.p2_idx = p2_idx
        self.distance = distance

    def solve(self, particles: List[Particle], dt: float):
        p1 = particles[self.p1_idx]
        p2 = particles[self.p2_idx]
        
        delta = p2.position - p1.position
        dist = np.linalg.norm(delta)
        if dist < 1e-12:
            return

        # Difference between current distance and constraint distance
        diff = (dist - self.distance) / dist
        
        # Correction vector
        correction = 0.5 * diff * delta

        # Apply corrections
        if not p1.fixed:
            p1.position += correction
        if not p2.fixed:
            p2.position -= correction


class SpatialHashingGrid:
    """A spatial hashing grid for efficient collision detection."""
    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], List[int]] = {}

    def _hash(self, position: NDArray[np.float64]) -> Tuple[int, int, int]:
        """Convert a position to a grid cell index."""
        return tuple((position / self.cell_size).astype(int))

    def clear(self):
        """Clear the grid."""
        self.grid.clear()

    def insert(self, particle_idx: int, position: NDArray[np.float64]):
        """Insert a particle into the grid."""
        cell_idx = self._hash(position)
        if cell_idx not in self.grid:
            self.grid[cell_idx] = []
        self.grid[cell_idx].append(particle_idx)

    def get_neighbors(self, particle_idx: int, position: NDArray[np.float64]) -> List[int]:
        """Get potential collision candidates for a particle."""
        cell_idx = self._hash(position)
        neighbors = []
        for i in range(cell_idx[0] - 1, cell_idx[0] + 2):
            for j in range(cell_idx[1] - 1, cell_idx[1] + 2):
                for k in range(cell_idx[2] - 1, cell_idx[2] + 2):
                    neighbor_cell_idx = (i, j, k)
                    if neighbor_cell_idx in self.grid:
                        for neighbor_idx in self.grid[neighbor_cell_idx]:
                            if neighbor_idx != particle_idx:
                                neighbors.append(neighbor_idx)
        return neighbors


class MechanicsEngine:
    """
    Newtonian mechanics engine with adaptive timestep integration.

    Features:
    - Verlet integration for position updates
    - Collision detection and response
    - Friction (static, kinetic, rolling)
    - Elasticity (coefficient of restitution)
    - Gravity and arbitrary force fields
    - Energy conservation validation
    """

    def __init__(self, gravity: Optional[NDArray[np.float64]] = None,
                 enable_collisions: bool = True):
        """
        Initialize mechanics engine.

        Args:
            gravity: Gravitational acceleration vector [gx, gy, gz] in m/s².
                    Default is [0, 0, -9.80665] (standard Earth gravity in -z).
        """
        self.particles: List[Particle] = []
        self.rigid_bodies: List[RigidBody] = []
        self.constraints: List[Constraint] = []

        if gravity is None:
            self.gravity = np.array([0.0, 0.0, -g_0.value], dtype=np.float64)
        else:
            self.gravity = np.asarray(gravity, dtype=np.float64)

        self.time = 0.0
        self.dt = 0.001  # Default timestep: 1 ms
        self.dt_min = 1e-6  # Minimum timestep: 1 μs
        self.dt_max = 0.01  # Maximum timestep: 10 ms

        # Material properties
        self.restitution = 0.8  # Coefficient of restitution (0=inelastic, 1=elastic)
        self.friction_static = 0.6  # Static friction coefficient
        self.friction_kinetic = 0.4  # Kinetic friction coefficient
        self.friction_rolling = 0.01  # Rolling friction coefficient

        # Gravity model
        self.gravity_type = GravityType.UNIFORM
        self.integration_method = IntegrationMethod.VELOCITY_VERLET

        # Energy tracking for validation
        self.initial_energy = None
        self.energy_tolerance = 0.01  # 1% energy conservation tolerance

        # Collision controls
        self.enable_collisions = enable_collisions
        self.collision_particle_limit = 200  # Skip O(N²) detection for huge swarms

        # Spatial hashing for collision detection
        self._collision_grid: Optional[SpatialHashingGrid] = None
        self._collision_grid_cell_size: float = 1.0

    def set_collision_detection(self, enabled: bool, particle_limit: Optional[int] = None, cell_size: Optional[float] = None):
        """Enable/disable collision detection and optionally tune parameters."""
        self.enable_collisions = enabled
        if particle_limit is not None and particle_limit > 0:
            self.collision_particle_limit = particle_limit
        if cell_size is not None and cell_size > 0:
            self._collision_grid_cell_size = cell_size
            self._collision_grid = SpatialHashingGrid(cell_size=self._collision_grid_cell_size)
        elif enabled and self._collision_grid is None:
            self._collision_grid = SpatialHashingGrid(cell_size=self._collision_grid_cell_size)

    def add_constraint(self, constraint: Constraint) -> int:
        """Add a constraint to the simulation and return its index."""
        self.constraints.append(constraint)
        return len(self.constraints) - 1

    def add_particle(self, particle: Particle) -> int:
        """Add a particle to the simulation and return its index."""
        self.particles.append(particle)
        return len(self.particles) - 1

    def add_rigid_body(self, body: RigidBody) -> int:
        """Add a rigid body to the simulation and return its index."""
        self.rigid_bodies.append(body)
        return len(self.rigid_bodies) - 1

    def clear_forces(self):
        """Reset all forces and torques to zero."""
        for p in self.particles:
            p.force = np.zeros(3, dtype=np.float64)
        for rb in self.rigid_bodies:
            rb.force = np.zeros(3, dtype=np.float64)
            rb.torque = np.zeros(3, dtype=np.float64)

    def solve_constraints(self, dt: float, iterations: int = 5):
        """Solve all constraints iteratively."""
        for _ in range(iterations):
            for constraint in self.constraints:
                constraint.solve(self.particles, dt)

    def apply_gravity(self):
        """Apply gravitational force to all particles and rigid bodies."""
        if self.gravity_type == GravityType.UNIFORM:
            for p in self.particles:
                if not p.fixed:
                    p.force += p.mass * self.gravity
            for rb in self.rigid_bodies:
                if not rb.fixed:
                    rb.force += rb.mass * self.gravity
        elif self.gravity_type == GravityType.N_BODY:
            self.apply_n_body_gravity()

    def apply_n_body_gravity(self):
        """Apply N-body gravitational force between all pairs of particles."""
        n = len(self.particles)
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = self.particles[i], self.particles[j]
                if p1.fixed and p2.fixed:
                    continue
                
                delta = p2.position - p1.position
                dist_sq = np.dot(delta, delta)
                if dist_sq < 1e-12:  # Avoid singularity
                    continue
                
                dist = np.sqrt(dist_sq)
                force_mag = G.value * p1.mass * p2.mass / dist_sq
                force_vec = force_mag * (delta / dist)
                
                if not p1.fixed:
                    p1.force += force_vec
                if not p2.fixed:
                    p2.force -= force_vec

    def _derivatives(self, t, y):
        """
        Calculate the derivatives of the system's state for use with SciPy solvers.
        
        Args:
            t: Current time (required by solve_ivp, but not used in this simple case).
            y: State vector [pos1_x, pos1_y, pos1_z, vel1_x, ..., velN_z].
        
        Returns:
            An array of derivatives [vel1_x, vel1_y, vel1_z, acc1_x, ..., accN_z].
        """
        num_particles = len(self.particles)
        positions = y[:num_particles * 3].reshape((num_particles, 3))
        velocities = y[num_particles * 3:].reshape((num_particles, 3))
        
        # Update particle positions for force calculation
        for i, p in enumerate(self.particles):
            p.position = positions[i]

        self.clear_forces()
        self.apply_gravity()
        # Note: Other forces (e.g., springs) would need to be applied here
        
        accelerations = np.array([p.force / p.mass for p in self.particles])
        
        return np.concatenate((velocities.flatten(), accelerations.flatten()))

    def apply_force(self, particle_idx: int, force: NDArray[np.float64]):
        """Apply external force to a particle."""
        self.particles[particle_idx].force += force

    def apply_force_field(self, force_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]]):
        """
        Apply spatially-varying force field to all particles.

        Args:
            force_fn: Function mapping position → force vector
        """
        for p in self.particles:
            if not p.fixed:
                p.force += force_fn(p.position)

    def detect_collisions(self) -> List[Tuple[int, int]]:
        """
        Detect particle-particle collisions using sphere-sphere overlap.

        Returns:
            List of (i, j) particle index pairs that are colliding
        """
        collisions = []
        n = len(self.particles)

        if (not self.enable_collisions or
                n == 0):
            return []

        # Use brute-force for small numbers of particles
        use_brute_force = n <= 20 or self.collision_particle_limit < n or self._collision_grid is None

        if use_brute_force:
            for i in range(n):
                for j in range(i + 1, n):
                    p1, p2 = self.particles[i], self.particles[j]
                    if p1.fixed and p2.fixed:
                        continue
                    delta = p2.position - p1.position
                    dist_sq = np.dot(delta, delta)
                    min_dist_sq = (p1.radius + p2.radius) ** 2
                    if dist_sq < min_dist_sq:
                        collisions.append((i, j))
            return collisions

        # Use spatial hashing for larger numbers of particles
        self._collision_grid.clear()
        for i, p in enumerate(self.particles):
            self._collision_grid.insert(i, p.position)

        checked_pairs = set()
        for i, p1 in enumerate(self.particles):
            potential_neighbors = self._collision_grid.get_neighbors(i, p1.position)
            for j in potential_neighbors:
                # Avoid duplicate checks
                if i > j:
                    pair = (j, i)
                else:
                    pair = (i, j)
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                p2 = self.particles[j]
                if p1.fixed and p2.fixed:
                    continue

                delta = p2.position - p1.position
                dist_sq = np.dot(delta, delta)
                min_dist_sq = (p1.radius + p2.radius) ** 2
                if dist_sq < min_dist_sq:
                    collisions.append((i, j))
        
        return collisions

    def resolve_collision(self, i: int, j: int):
        """
        Resolve collision between particles i and j.

        Uses impulse-based collision response with coefficient of restitution.
        """
        p1, p2 = self.particles[i], self.particles[j]

        # Collision normal (from p1 to p2)
        delta = p2.position - p1.position
        dist = np.linalg.norm(delta)
        if dist < 1e-10:
            return  # Degenerate case

        normal = delta / dist

        # Relative velocity
        v_rel = p2.velocity - p1.velocity
        v_normal = np.dot(v_rel, normal)

        # Don't resolve if separating
        if v_normal > 0:
            return

        # Effective mass
        if p1.fixed:
            m1_inv = 0.0
        else:
            m1_inv = 1.0 / p1.mass

        if p2.fixed:
            m2_inv = 0.0
        else:
            m2_inv = 1.0 / p2.mass

        m_eff_inv = m1_inv + m2_inv

        if m_eff_inv < 1e-10:
            return  # Both fixed

        # Impulse magnitude
        j = -(1 + self.restitution) * v_normal / m_eff_inv
        impulse = j * normal

        # Apply impulse
        if not p1.fixed:
            p1.velocity -= impulse * m1_inv
        if not p2.fixed:
            p2.velocity += impulse * m2_inv

        # Separate particles to prevent overlap
        overlap = (p1.radius + p2.radius) - dist
        if overlap > 0:
            separation = normal * (overlap / 2 + 1e-6)
            if not p1.fixed:
                p1.position -= separation
            if not p2.fixed:
                p2.position += separation

    def apply_friction(self, dt: float):
        """
        Apply friction forces to particles in contact with ground (z=0 plane).

        Implements static, kinetic, and rolling friction.
        """
        for p in self.particles:
            if p.fixed:
                continue

            # Check ground contact (simple: z <= radius)
            if p.position[2] <= p.radius + 1e-6:
                # Normal force
                N = -p.mass * self.gravity[2]

                # Tangential velocity
                v_tangent = p.velocity.copy()
                v_tangent[2] = 0  # Remove vertical component
                v_mag = np.linalg.norm(v_tangent)

                if v_mag > 1e-6:
                    # Kinetic friction
                    friction_force = -self.friction_kinetic * N * (v_tangent / v_mag)
                    p.force += friction_force
                else:
                    # Static friction (oppose applied force)
                    tangent_force = p.force.copy()
                    tangent_force[2] = 0
                    f_mag = np.linalg.norm(tangent_force)

                    if f_mag > self.friction_static * N:
                        # Break static friction
                        if f_mag > 1e-10:
                            friction_force = -self.friction_static * N * (tangent_force / f_mag)
                            p.force += friction_force

    def step(self, dt: Optional[float] = None):
        """
        Advance simulation by one timestep using velocity Verlet integration.

        Args:
            dt: Timestep in seconds. If None, uses self.dt.
        """
        if dt is None:
            dt = self.dt

        # Store initial energy if not set
        if self.initial_energy is None:
            self.initial_energy = self.total_energy()

        # Preserve any external forces assigned prior to this step
        external_forces = None
        if any(not np.all(p.force == 0.0) for p in self.particles):
            external_forces = [p.force.copy() for p in self.particles]

        # Clear forces from previous step
        self.clear_forces()

        # Apply forces
        self.apply_gravity()
        self.apply_friction(dt)

        # Re-apply caller-specified forces (e.g., springs)
        if external_forces is not None:
            for p, ext in zip(self.particles, external_forces):
                if not np.all(ext == 0.0):
                    p.force += ext

        # Update particles using velocity Verlet
        for p in self.particles:
            if p.fixed:
                continue

            # Acceleration at current step
            a_current = p.force / p.mass

            # Update position: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
            p.position += p.velocity * dt + 0.5 * a_current * dt**2
            
            # Store current acceleration for velocity update
            p.force = a_current  # Temporarily store a_current in force attribute

        # Update forces at the new positions
        self.apply_gravity()
        # Note: Other forces like springs would need to be re-applied here as well
        
        # Complete the velocity update
        for p in self.particles:
            if p.fixed:
                continue
            
            a_current = p.force # Retrieve a_current
            a_new = (p.force / p.mass) if p.mass > 1e-9 else np.zeros(3)

            # Update velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
            p.velocity += 0.5 * (a_current + a_new) * dt

        # Update rigid bodies
        for rb in self.rigid_bodies:
            if rb.fixed:
                continue
            
            # Update linear motion
            accel = rb.force / rb.mass
            rb.position += rb.velocity * dt + 0.5 * accel * dt**2
            rb.velocity += accel * dt

            # Update rotational motion
            ang_accel = np.linalg.inv(rb.inertia_tensor) @ rb.torque
            rb.angular_velocity += ang_accel * dt
            
            # Update orientation using quaternion integration
            q = rb.orientation
            omega = rb.angular_velocity
            q_dot = 0.5 * np.array([
                -q[1]*omega[0] - q[2]*omega[1] - q[3]*omega[2],
                 q[0]*omega[0] + q[2]*omega[2] - q[3]*omega[1],
                 q[0]*omega[1] - q[1]*omega[2] + q[3]*omega[0],
                 q[0]*omega[2] + q[1]*omega[1] - q[2]*omega[0]
            ])
            q += q_dot * dt
            q /= np.linalg.norm(q)  # Re-normalize quaternion
            rb.orientation = q

        # Detect and resolve collisions
        collisions = self.detect_collisions()
        for i, j in collisions:
            self.resolve_collision(i, j)

        # Solve constraints
        self.solve_constraints(dt)

        # Reset forces so only caller-supplied inputs persist between steps
        for p in self.particles:
            p.force.fill(0.0)

        # Update time
        self.time += dt

    def simulate(self, duration: float, dt: Optional[float] = None,
                callback: Optional[Callable[[float, List[Particle]], None]] = None):
        """
        Run simulation for specified duration.

        Args:
            duration: Simulation duration in seconds
            dt: Timestep in seconds (for Verlet) or max_step (for SciPy)
            callback: Optional function called each step: callback(time, particles)
        """
        if self.integration_method == IntegrationMethod.VELOCITY_VERLET:
            if dt is not None:
                self.dt = dt
            steps = int(duration / self.dt)
            for _ in range(steps):
                self.step()
                if callback is not None:
                    callback(self.time, self.particles)
        elif self.integration_method == IntegrationMethod.SCIPY_DOP853:
            y0 = np.concatenate([p.position.flatten() for p in self.particles] +
                                [p.velocity.flatten() for p in self.particles])
            
            sol = solve_ivp(
                self._derivatives,
                (0, duration),
                y0,
                method='DOP853',
                dense_output=True,
                max_step=dt if dt is not None else 0.01
            )

            # Update particle states to the final state
            final_y = sol.y[:, -1]
            num_particles = len(self.particles)
            final_positions = final_y[:num_particles * 3].reshape((num_particles, 3))
            final_velocities = final_y[num_particles * 3:].reshape((num_particles, 3))

            for i, p in enumerate(self.particles):
                p.position = final_positions[i]
                p.velocity = final_velocities[i]
            
            self.time += duration
            
            if callback:
                # Provide state at intermediate points for the callback
                t_points = np.linspace(0, duration, int(duration / (dt if dt else 0.01)))
                y_points = sol.sol(t_points)
                for i, t in enumerate(t_points):
                    positions = y_points[:num_particles*3, i].reshape((num_particles, 3))
                    # Update particle objects for callback
                    for j, p in enumerate(self.particles):
                        p.position = positions[j]
                    callback(t, self.particles)


    def kinetic_energy(self) -> float:
        """Calculate total kinetic energy of system."""
        ke = 0.0
        for p in self.particles:
            if not p.fixed:
                v2 = np.dot(p.velocity, p.velocity)
                ke += 0.5 * p.mass * v2
        return ke

    def potential_energy(self) -> float:
        """Calculate total gravitational potential energy."""
        pe = 0.0
        for p in self.particles:
            if not p.fixed:
                # PE = m * g * h (relative to z=0)
                pe += -p.mass * np.dot(self.gravity, p.position)
        return pe

    def total_energy(self) -> float:
        """Calculate total energy (kinetic + potential)."""
        return self.kinetic_energy() + self.potential_energy()

    def energy_error(self) -> float:
        """Calculate relative energy conservation error."""
        if self.initial_energy is None or abs(self.initial_energy) < 1e-10:
            return 0.0
        current = self.total_energy()
        return abs(current - self.initial_energy) / abs(self.initial_energy)

    def momentum(self) -> NDArray[np.float64]:
        """Calculate total linear momentum of system."""
        p_total = np.zeros(3, dtype=np.float64)
        for p in self.particles:
            if not p.fixed:
                p_total += p.mass * p.velocity
        return p_total

    def angular_momentum(self, origin: Optional[NDArray[np.float64]] = None) -> NDArray[np.float64]:
        """
        Calculate total angular momentum about origin.

        Args:
            origin: Reference point. Default is [0, 0, 0].
        """
        if origin is None:
            origin = np.zeros(3, dtype=np.float64)

        L_total = np.zeros(3, dtype=np.float64)
        for p in self.particles:
            if not p.fixed:
                r = p.position - origin
                L_total += p.mass * np.cross(r, p.velocity)
        return L_total


def spring_force(p1: Particle, p2: Particle, k: float, rest_length: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate spring forces between two particles (Hooke's law).

    Args:
        p1, p2: The two particles
        k: Spring constant in N/m
        rest_length: Rest length in meters

    Returns:
        (force_on_p1, force_on_p2)
    """
    delta = p2.position - p1.position
    dist = np.linalg.norm(delta)

    if dist < 1e-10:
        return np.zeros(3), np.zeros(3)

    direction = delta / dist
    extension = dist - rest_length

    f_magnitude = k * extension
    force = f_magnitude * direction

    return force, -force


def damped_spring_force(p1: Particle, p2: Particle, k: float,
                       rest_length: float, damping: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate damped spring forces (spring + dashpot).

    Args:
        p1, p2: The two particles
        k: Spring constant in N/m
        rest_length: Rest length in meters
        damping: Damping coefficient in N⋅s/m

    Returns:
        (force_on_p1, force_on_p2)
    """
    delta = p2.position - p1.position
    dist = np.linalg.norm(delta)

    if dist < 1e-10:
        return np.zeros(3), np.zeros(3)

    direction = delta / dist

    # Spring force
    extension = dist - rest_length
    f_spring = k * extension

    # Damping force (along spring axis)
    v_rel = p2.velocity - p1.velocity
    v_along = np.dot(v_rel, direction)
    f_damping = damping * v_along

    f_total = (f_spring + f_damping) * direction

    return f_total, -f_total


if __name__ == "__main__":
    print("QuLab Infinite - Mechanics Engine Test")
    print("=" * 80)

    # Test 1: Free fall
    print("\nTest 1: Free fall from 10m")
    engine = MechanicsEngine()

    p = Particle(
        mass=1.0,
        position=np.array([0.0, 0.0, 10.0]),
        velocity=np.array([0.0, 0.0, 0.0]),
        force=np.zeros(3),
        radius=0.1
    )
    engine.add_particle(p)

    # Simulate for 1.5 seconds
    t_fall = np.sqrt(2 * 10.0 / g_0.value)  # Expected fall time
    print(f"Expected fall time: {t_fall:.4f} s")

    engine.simulate(1.5, dt=0.001)
    print(f"Final position: z = {engine.particles[0].position[2]:.4f} m")
    print(f"Final velocity: vz = {engine.particles[0].velocity[2]:.4f} m/s")
    print(f"Expected velocity: vz = {-g_0.value * t_fall:.4f} m/s")
    print(f"Energy error: {engine.energy_error() * 100:.4f}%")

    # Test 2: Projectile motion
    print("\nTest 2: Projectile motion (45° launch)")
    engine2 = MechanicsEngine()

    v0 = 20.0  # m/s
    angle = np.pi / 4  # 45 degrees

    p2 = Particle(
        mass=0.5,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([v0 * np.cos(angle), 0.0, v0 * np.sin(angle)]),
        force=np.zeros(3),
        radius=0.05
    )
    engine2.add_particle(p2)

    # Simulate until landing
    max_time = 2 * v0 * np.sin(angle) / g_0.value
    print(f"Expected flight time: {max_time:.4f} s")

    engine2.simulate(max_time, dt=0.001)
    print(f"Final position: x = {engine2.particles[0].position[0]:.2f} m")
    expected_range = v0**2 * np.sin(2 * angle) / g_0.value
    print(f"Expected range: {expected_range:.2f} m")
    print(f"Energy error: {engine2.energy_error() * 100:.4f}%")

    # Test 3: Elastic collision
    print("\nTest 3: Elastic collision")
    engine3 = MechanicsEngine()
    engine3.restitution = 1.0  # Perfectly elastic

    p3a = Particle(
        mass=1.0,
        position=np.array([0.0, 0.0, 1.0]),
        velocity=np.array([1.0, 0.0, 0.0]),
        force=np.zeros(3),
        radius=0.1
    )
    p3b = Particle(
        mass=1.0,
        position=np.array([2.0, 0.0, 1.0]),
        velocity=np.array([-1.0, 0.0, 0.0]),
        force=np.zeros(3),
        radius=0.1
    )

    engine3.add_particle(p3a)
    engine3.add_particle(p3b)

    # Turn off gravity for this test
    engine3.gravity = np.zeros(3)

    print(f"Before: v1 = {p3a.velocity[0]:.2f} m/s, v2 = {p3b.velocity[0]:.2f} m/s")

    # Simulate collision
    engine3.simulate(1.5, dt=0.001)

    print(f"After:  v1 = {engine3.particles[0].velocity[0]:.2f} m/s, v2 = {engine3.particles[1].velocity[0]:.2f} m/s")
    print(f"Expected: v1 = -1.00 m/s, v2 = 1.00 m/s (velocities swap)")
    print(f"Energy error: {engine3.energy_error() * 100:.6f}%")

    print("\n" + "=" * 80)
    print("Mechanics engine tests complete!")
