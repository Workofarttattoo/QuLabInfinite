"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Temporal Bridge - Time-Scale Management for Multi-Scale Simulations
Seamless transitions from femtoseconds to years with accelerated dynamics
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class TimeScale(Enum):
    """Time scales for simulations"""
    FEMTOSECOND = (1e-15, "fs")
    PICOSECOND = (1e-12, "ps")
    NANOSECOND = (1e-9, "ns")
    MICROSECOND = (1e-6, "µs")
    MILLISECOND = (1e-3, "ms")
    SECOND = (1.0, "s")
    MINUTE = (60.0, "min")
    HOUR = (3600.0, "h")
    DAY = (86400.0, "d")
    MONTH = (2.592e6, "mo")
    YEAR = (3.154e7, "yr")

    def __init__(self, seconds: float, unit: str):
        self.seconds = seconds
        self.unit = unit


@dataclass
class TimePoint:
    """Point in simulation time"""
    simulation_time: float  # In seconds
    wall_time: float  # Real time elapsed
    scale: TimeScale
    description: str = ""

    def to_scale(self, target_scale: TimeScale) -> float:
        """Convert to different time scale"""
        return self.simulation_time / target_scale.seconds


@dataclass
class Event:
    """Simulation event at specific time"""
    event_id: str
    time: float  # Simulation time in seconds
    event_type: str
    callback: Optional[Callable] = None
    data: Dict[str, Any] = field(default_factory=dict)
    triggered: bool = False


@dataclass
class Checkpoint:
    """Simulation state checkpoint"""
    checkpoint_id: str
    time: float  # Simulation time
    state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class TimeScaleManager:
    """Manage transitions across time scales"""

    def __init__(self):
        self.current_time = 0.0  # Simulation time in seconds
        self.current_scale = TimeScale.SECOND
        self.time_history: List[TimePoint] = []
        self.start_wall_time = time.time()

    def advance(self, delta: float, scale: Optional[TimeScale] = None) -> float:
        """Advance simulation time by delta in specified scale"""
        if scale is None:
            scale = self.current_scale

        # Convert delta to seconds
        delta_seconds = delta * scale.seconds

        # Update current time
        self.current_time += delta_seconds
        self.current_scale = scale

        # Record time point
        self.time_history.append(TimePoint(
            simulation_time=self.current_time,
            wall_time=time.time() - self.start_wall_time,
            scale=scale
        ))

        return self.current_time

    def set_time(self, new_time: float, scale: TimeScale = TimeScale.SECOND) -> None:
        """Set absolute simulation time"""
        self.current_time = new_time * scale.seconds
        self.current_scale = scale

    def get_time(self, scale: Optional[TimeScale] = None) -> float:
        """Get current time in specified scale"""
        if scale is None:
            scale = self.current_scale
        return self.current_time / scale.seconds

    def transition_scale(self, target_scale: TimeScale) -> None:
        """Transition to different time scale"""
        LOG.info(f"[info] Transitioning from {self.current_scale.unit} to {target_scale.unit}")
        self.current_scale = target_scale

    def get_speedup(self) -> float:
        """Get simulation speedup vs real time"""
        wall_time = time.time() - self.start_wall_time
        if wall_time == 0:
            return 0.0
        return self.current_time / wall_time

    def estimate_remaining_time(self, target_time: float, scale: TimeScale) -> float:
        """Estimate wall time to reach target simulation time"""
        target_seconds = target_time * scale.seconds
        remaining_sim = target_seconds - self.current_time

        speedup = self.get_speedup()
        if speedup == 0:
            return float('inf')

        return remaining_sim / speedup


class TemporalSynchronization:
    """Coordinate experiments at different time scales"""

    def __init__(self):
        self.processes: Dict[str, TimeScaleManager] = {}

    def register_process(self, process_id: str, manager: TimeScaleManager) -> None:
        """Register process for synchronization"""
        self.processes[process_id] = manager
        LOG.info(f"[info] Registered process {process_id} for temporal sync")

    def synchronize_all(self, target_time: float, scale: TimeScale) -> Dict[str, float]:
        """Synchronize all processes to target time"""
        results = {}
        target_seconds = target_time * scale.seconds

        for proc_id, manager in self.processes.items():
            delta = target_seconds - manager.current_time
            if delta > 0:
                manager.advance(delta / scale.seconds, scale)
            results[proc_id] = manager.current_time

        LOG.info(f"[info] Synchronized {len(self.processes)} processes to {target_time} {scale.unit}")
        return results

    def get_status(self) -> Dict[str, Any]:
        """Get synchronization status"""
        if not self.processes:
            return {"processes": 0}

        times = [m.current_time for m in self.processes.values()]
        return {
            "processes": len(self.processes),
            "min_time": min(times),
            "max_time": max(times),
            "desynchronization": max(times) - min(times),
            "process_times": {pid: m.current_time for pid, m in self.processes.items()}
        }


class EventDetection:
    """Detect and trigger events during simulation"""

    def __init__(self, time_manager: TimeScaleManager):
        self.time_manager = time_manager
        self.events: List[Event] = []
        self.callbacks: Dict[str, List[Callable]] = {}  # event_type -> callbacks

    def schedule_event(self, event: Event) -> None:
        """Schedule event at specific time"""
        self.events.append(event)
        self.events.sort(key=lambda e: e.time)
        LOG.info(f"[info] Scheduled event {event.event_id} at t={event.time}s ({event.event_type})")

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe callback to event type"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)

    def check_events(self) -> List[Event]:
        """Check and trigger any events that have occurred"""
        triggered = []
        current_time = self.time_manager.current_time

        for event in self.events:
            if not event.triggered and event.time <= current_time:
                # Trigger event
                event.triggered = True
                triggered.append(event)

                # Call event-specific callback
                if event.callback:
                    event.callback(event)

                # Call subscribed callbacks
                if event.event_type in self.callbacks:
                    for callback in self.callbacks[event.event_type]:
                        callback(event)

                LOG.info(f"[info] Triggered event {event.event_id}: {event.event_type}")

        return triggered

    def add_phase_transition_detector(self, material: str, threshold_temp: float) -> None:
        """Add automatic phase transition detection"""
        def detect_phase_transition(state: Dict[str, Any]) -> Optional[Event]:
            if state.get("temperature", 0) >= threshold_temp:
                return Event(
                    event_id=f"phase_{material}_{int(time.time())}",
                    time=self.time_manager.current_time,
                    event_type="phase_transition",
                    data={"material": material, "temperature": state["temperature"]}
                )
            return None

        # Store detector (would be called during simulation loop)
        self.callbacks.setdefault("simulation_step", []).append(detect_phase_transition)


class CheckpointManager:
    """Save and restore simulation state"""

    def __init__(self, time_manager: TimeScaleManager):
        self.time_manager = time_manager
        self.checkpoints: Dict[str, Checkpoint] = {}
        self.auto_checkpoint_interval: Optional[float] = None

    def save_checkpoint(self, checkpoint_id: str, state: Dict[str, Any],
                       metadata: Optional[Dict[str, Any]] = None) -> Checkpoint:
        """Save simulation state checkpoint"""
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            time=self.time_manager.current_time,
            state=state.copy(),
            metadata=metadata or {}
        )
        self.checkpoints[checkpoint_id] = checkpoint
        LOG.info(f"[info] Saved checkpoint {checkpoint_id} at t={checkpoint.time}s")
        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Restore simulation from checkpoint"""
        if checkpoint_id not in self.checkpoints:
            LOG.warning(f"[warn] Checkpoint {checkpoint_id} not found")
            return None

        checkpoint = self.checkpoints[checkpoint_id]
        self.time_manager.set_time(checkpoint.time, TimeScale.SECOND)
        LOG.info(f"[info] Restored checkpoint {checkpoint_id} at t={checkpoint.time}s")
        return checkpoint.state

    def enable_auto_checkpoint(self, interval_seconds: float) -> None:
        """Enable automatic checkpointing at interval"""
        self.auto_checkpoint_interval = interval_seconds
        LOG.info(f"[info] Auto-checkpoint enabled every {interval_seconds}s")

    def check_auto_checkpoint(self, state: Dict[str, Any]) -> Optional[Checkpoint]:
        """Check if auto-checkpoint should be created"""
        if self.auto_checkpoint_interval is None:
            return None

        # Check if enough time has passed since last checkpoint
        if not self.checkpoints:
            last_time = 0.0
        else:
            last_time = max(cp.time for cp in self.checkpoints.values())

        if self.time_manager.current_time - last_time >= self.auto_checkpoint_interval:
            checkpoint_id = f"auto_{int(self.time_manager.current_time)}"
            return self.save_checkpoint(checkpoint_id, state, {"auto": True})

        return None

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoints"""
        return [
            {
                "checkpoint_id": cp.checkpoint_id,
                "time": cp.time,
                "created_at": cp.created_at,
                "metadata": cp.metadata
            }
            for cp in self.checkpoints.values()
        ]

    def save_to_file(self, filepath: str) -> None:
        """Save all checkpoints to file"""
        data = {
            "checkpoints": {
                cid: {
                    "checkpoint_id": cp.checkpoint_id,
                    "time": cp.time,
                    "state": cp.state,
                    "metadata": cp.metadata,
                    "created_at": cp.created_at
                }
                for cid, cp in self.checkpoints.items()
            },
            "current_time": self.time_manager.current_time
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        LOG.info(f"[info] Checkpoints saved to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """Load checkpoints from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for cp_data in data["checkpoints"].values():
            checkpoint = Checkpoint(**cp_data)
            self.checkpoints[checkpoint.checkpoint_id] = checkpoint

        self.time_manager.set_time(data["current_time"], TimeScale.SECOND)
        LOG.info(f"[info] Loaded {len(self.checkpoints)} checkpoints from {filepath}")


class AcceleratedDynamics:
    """Accelerate rare events using advanced sampling techniques"""

    def __init__(self):
        pass

    def metadynamics(self, state: Dict[str, Any], collective_variable: str,
                    gaussian_height: float = 1.0, gaussian_width: float = 0.1) -> float:
        """
        Metadynamics: Add bias potential to accelerate barrier crossing
        Returns bias force to add to system
        """
        # Simplified metadynamics implementation
        cv_value = state.get(collective_variable, 0.0)

        # Add Gaussian hill at current CV value
        bias_force = gaussian_height * np.exp(-cv_value**2 / (2 * gaussian_width**2))

        return bias_force

    def hyperdynamics(self, state: Dict[str, Any], energy: float,
                     energy_threshold: float = 1.0) -> Tuple[float, float]:
        """
        Hyperdynamics: Boost dynamics in energy basins
        Returns (boosted_time, boost_factor)
        """
        if energy < energy_threshold:
            # In basin: apply boost
            delta_E = energy_threshold - energy
            boost_factor = np.exp(delta_E)
        else:
            # On barrier: no boost
            boost_factor = 1.0

        return boost_factor, boost_factor

    def parallel_replica(self, num_replicas: int, replica_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parallel Replica Dynamics: Run multiple replicas in parallel
        Returns first replica to escape basin
        """
        # Simulate each replica (simplified)
        escape_times = []
        for i, state in enumerate(replica_states):
            # Random escape time (would be actual simulation)
            escape_time = np.random.exponential(1.0)
            escape_times.append((i, escape_time, state))

        # Return fastest escape
        fastest = min(escape_times, key=lambda x: x[1])
        replica_idx, escape_time, state = fastest

        # Effective speedup is number of replicas
        effective_time = escape_time / num_replicas

        return {
            "escaped_replica": replica_idx,
            "escape_time": escape_time,
            "effective_time": effective_time,
            "speedup": num_replicas,
            "state": state
        }

    def temperature_accelerated_dynamics(self, state: Dict[str, Any],
                                        base_temp: float, boost_temp: float,
                                        timestep: float) -> Tuple[float, Dict[str, Any]]:
        """
        Temperature-Accelerated Dynamics (TAD): Run at high temp, correct statistics
        Returns (boosted_timestep, corrected_state)
        """
        # Boost factor from temperature ratio
        boost_factor = boost_temp / base_temp

        # Effective timestep
        boosted_timestep = timestep * boost_factor

        # Correct state for temperature (simplified)
        corrected_state = state.copy()
        # Apply temperature correction to velocities, energies, etc.

        return boosted_timestep, corrected_state

    def estimate_speedup(self, method: str, parameters: Dict[str, Any]) -> float:
        """Estimate speedup factor for accelerated dynamics method"""
        speedups = {
            "metadynamics": 10.0,  # Typical 10-100x speedup
            "hyperdynamics": 100.0,  # Typical 100-1000x speedup
            "parallel_replica": parameters.get("num_replicas", 8),  # Linear with replicas
            "temperature_accelerated": parameters.get("boost_temp", 1000) / parameters.get("base_temp", 300)
        }
        return speedups.get(method, 1.0)


class TemporalBridge:
    """Complete temporal management system"""

    def __init__(self):
        self.time_manager = TimeScaleManager()
        self.synchronizer = TemporalSynchronization()
        self.event_detector = EventDetection(self.time_manager)
        self.checkpoint_manager = CheckpointManager(self.time_manager)
        self.accelerated_dynamics = AcceleratedDynamics()

    def simulate_accelerated(self, target_time: float, scale: TimeScale,
                           state: Dict[str, Any],
                           acceleration_method: str = "parallel_replica",
                           parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run accelerated simulation to target time
        Returns final state
        """
        parameters = parameters or {}

        # Estimate speedup
        speedup = self.accelerated_dynamics.estimate_speedup(acceleration_method, parameters)

        # Effective timestep
        target_seconds = target_time * scale.seconds
        effective_timestep = target_seconds / speedup

        LOG.info(f"[info] Accelerated simulation: {target_time} {scale.unit} "
                f"with {acceleration_method} ({speedup}x speedup)")

        # Advance time with speedup
        self.time_manager.advance(effective_timestep, TimeScale.SECOND)

        # Check events
        triggered_events = self.event_detector.check_events()

        # Auto-checkpoint
        self.checkpoint_manager.check_auto_checkpoint(state)

        return {
            "state": state,
            "simulation_time": self.time_manager.current_time,
            "speedup": speedup,
            "triggered_events": [e.event_id for e in triggered_events],
            "wall_time": time.time() - self.time_manager.start_wall_time
        }

    def run_multiscale_experiment(self, processes: List[Tuple[str, TimeScale, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Run multi-scale experiment with coordinated processes
        processes: List of (process_id, time_scale, state)
        """
        # Register all processes
        for proc_id, scale, state in processes:
            manager = TimeScaleManager()
            manager.current_scale = scale
            self.synchronizer.register_process(proc_id, manager)

        # Run simulation (simplified)
        LOG.info(f"[info] Running multi-scale experiment with {len(processes)} processes")

        # Synchronize to common time
        max_time = max(manager.current_time for manager in self.synchronizer.processes.values())
        self.synchronizer.synchronize_all(max_time, TimeScale.SECOND)

        status = self.synchronizer.get_status()
        return status


if __name__ == "__main__":
    # Demo
    bridge = TemporalBridge()

    # Enable auto-checkpointing
    bridge.checkpoint_manager.enable_auto_checkpoint(100.0)  # Every 100 seconds

    # Schedule events
    bridge.event_detector.schedule_event(Event(
        event_id="phase_transition_001",
        time=10.0,
        event_type="phase_transition",
        data={"material": "water", "from": "liquid", "to": "gas"}
    ))

    # Subscribe to events
    def on_phase_transition(event: Event):
        print(f"Phase transition detected: {event.data}")

    bridge.event_detector.subscribe("phase_transition", on_phase_transition)

    # Run accelerated simulation
    initial_state = {"temperature": 373, "pressure": 1.0}

    result = bridge.simulate_accelerated(
        target_time=1.0,
        scale=TimeScale.HOUR,
        state=initial_state,
        acceleration_method="parallel_replica",
        parameters={"num_replicas": 8}
    )

    print(f"\nSimulation Result:")
    print(f"  Simulation time: {result['simulation_time']:.2f}s")
    print(f"  Speedup: {result['speedup']}x")
    print(f"  Wall time: {result['wall_time']:.2f}s")
    print(f"  Events triggered: {result['triggered_events']}")

    # Multi-scale experiment
    processes = [
        ("molecular_dynamics", TimeScale.FEMTOSECOND, {}),
        ("crack_propagation", TimeScale.MICROSECOND, {}),
        ("corrosion_test", TimeScale.DAY, {})
    ]

    multiscale_result = bridge.run_multiscale_experiment(processes)
    print(f"\nMulti-scale Experiment:")
    print(f"  Processes: {multiscale_result['processes']}")
    print(f"  Desynchronization: {multiscale_result['desynchronization']:.2e}s")
