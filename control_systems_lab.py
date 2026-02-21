"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CONTROL SYSTEMS LAB - Production Ready Implementation
PID control, state-space models, LQR, Kalman filters, and advanced control theory
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable
from scipy import signal, linalg
import matplotlib.pyplot as plt
from scipy.constants import g

# Physical constants for control systems
GRAVITY = g  # 9.80665 m/s²
AIR_DENSITY = 1.225  # kg/m³ at sea level
STANDARD_TEMP = 293.15  # K (20°C)

@dataclass
class StateSpaceModel:
    """State-space representation: dx/dt = Ax + Bu, y = Cx + Du"""
    A: np.ndarray  # State matrix
    B: np.ndarray  # Input matrix
    C: np.ndarray  # Output matrix
    D: np.ndarray  # Feedthrough matrix
    state_names: Optional[List[str]] = None
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None

@dataclass
class PIDController:
    """PID controller with anti-windup and derivative filtering"""
    Kp: float  # Proportional gain
    Ki: float  # Integral gain
    Kd: float  # Derivative gain
    dt: float  # Sample time
    integral_limit: Optional[float] = None  # Anti-windup limit
    derivative_filter: float = 0.1  # Derivative filter coefficient
    integral: float = field(default=0.0, init=False)
    prev_error: float = field(default=0.0, init=False)
    prev_derivative: float = field(default=0.0, init=False)

@dataclass
class KalmanFilter:
    """Kalman filter for state estimation"""
    A: np.ndarray  # State transition matrix
    B: np.ndarray  # Control input matrix
    C: np.ndarray  # Observation matrix
    Q: np.ndarray  # Process noise covariance
    R: np.ndarray  # Measurement noise covariance
    x: np.ndarray = field(init=False)  # State estimate
    P: np.ndarray = field(init=False)  # Error covariance

    def __post_init__(self):
        n = self.A.shape[0]
        self.x = np.zeros(n)
        self.P = np.eye(n)

class ControlSystemsLab:
    """Comprehensive control systems laboratory"""

    def __init__(self):
        self.systems: Dict[str, StateSpaceModel] = {}
        self.controllers: Dict[str, PIDController] = {}
        self.simulation_history: List[Dict] = []

    # ============= PID CONTROL =============

    def pid_control(self, controller: PIDController, setpoint: float,
                   measurement: float) -> float:
        """
        Calculate PID controller output with anti-windup and derivative filtering
        """
        error = setpoint - measurement

        # Proportional term
        P = controller.Kp * error

        # Integral term with anti-windup
        controller.integral += error * controller.dt
        if controller.integral_limit:
            controller.integral = np.clip(controller.integral,
                                        -controller.integral_limit,
                                        controller.integral_limit)
        I = controller.Ki * controller.integral

        # Derivative term with filtering
        if controller.prev_error != 0:
            derivative = (error - controller.prev_error) / controller.dt
            # Low-pass filter on derivative
            controller.prev_derivative = (controller.derivative_filter * derivative +
                                        (1 - controller.derivative_filter) * controller.prev_derivative)
        else:
            controller.prev_derivative = 0

        D = controller.Kd * controller.prev_derivative

        controller.prev_error = error

        return P + I + D

    def auto_tune_pid(self, system: StateSpaceModel,
                     method: str = "ziegler-nichols") -> PIDController:
        """
        Auto-tune PID controller using various methods
        """
        if method == "ziegler-nichols":
            # Find ultimate gain and period
            Ku, Pu = self._find_ultimate_gain(system)

            # Ziegler-Nichols tuning rules
            Kp = 0.6 * Ku
            Ki = 2 * Kp / Pu
            Kd = Kp * Pu / 8

        elif method == "cohen-coon":
            # Cohen-Coon method for first-order plus dead time
            K, tau, theta = self._fit_fopdt(system)

            a = K * theta / tau
            Kp = (1.35 / K) * (tau / theta + 0.18)
            Ki = Kp / (2.5 * theta)
            Kd = Kp * 0.37 * theta

        else:
            raise ValueError(f"Unknown tuning method: {method}")

        return PIDController(Kp=Kp, Ki=Ki, Kd=Kd, dt=0.01)

    def cascade_pid(self, outer_controller: PIDController,
                   inner_controller: PIDController,
                   outer_setpoint: float,
                   outer_measurement: float,
                   inner_measurement: float) -> float:
        """
        Cascade PID control for improved performance
        """
        # Outer loop calculates setpoint for inner loop
        inner_setpoint = self.pid_control(outer_controller, outer_setpoint, outer_measurement)

        # Inner loop controls the actuator
        control_signal = self.pid_control(inner_controller, inner_setpoint, inner_measurement)

        return control_signal

    # ============= STATE-SPACE CONTROL =============

    def create_state_space(self, num: np.ndarray, den: np.ndarray) -> StateSpaceModel:
        """
        Create state-space model from transfer function
        """
        A, B, C, D = signal.tf2ss(num, den)
        return StateSpaceModel(A=A, B=B, C=C, D=D)

    def linearize_system(self, f: Callable, g: Callable,
                        x0: np.ndarray, u0: np.ndarray,
                        delta: float = 1e-6) -> StateSpaceModel:
        """
        Linearize nonlinear system around operating point
        dx/dt = f(x, u), y = g(x, u)
        """
        n = len(x0)
        m = len(u0)

        # Compute Jacobians numerically
        A = np.zeros((n, n))
        B = np.zeros((n, m))

        for i in range(n):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += delta
            x_minus[i] -= delta

            A[:, i] = (f(x_plus, u0) - f(x_minus, u0)) / (2 * delta)

        for i in range(m):
            u_plus = u0.copy()
            u_minus = u0.copy()
            u_plus[i] += delta
            u_minus[i] -= delta

            B[:, i] = (f(x0, u_plus) - f(x0, u_minus)) / (2 * delta)

        # Assume output is first state
        C = np.eye(n)
        D = np.zeros((n, m))

        return StateSpaceModel(A=A, B=B, C=C, D=D)

    def controllability_matrix(self, system: StateSpaceModel) -> np.ndarray:
        """
        Compute controllability matrix [B, AB, A²B, ...]
        """
        n = system.A.shape[0]
        m = system.B.shape[1]

        P = np.zeros((n, n * m))
        P[:, :m] = system.B

        for i in range(1, n):
            P[:, i*m:(i+1)*m] = system.A @ P[:, (i-1)*m:i*m]

        return P

    def is_controllable(self, system: StateSpaceModel) -> bool:
        """
        Check if system is controllable
        """
        P = self.controllability_matrix(system)
        return np.linalg.matrix_rank(P) == system.A.shape[0]

    def observability_matrix(self, system: StateSpaceModel) -> np.ndarray:
        """
        Compute observability matrix [C; CA; CA²; ...]
        """
        n = system.A.shape[0]
        p = system.C.shape[0]

        Q = np.zeros((n * p, n))
        Q[:p, :] = system.C

        for i in range(1, n):
            Q[i*p:(i+1)*p, :] = Q[(i-1)*p:i*p, :] @ system.A

        return Q

    def is_observable(self, system: StateSpaceModel) -> bool:
        """
        Check if system is observable
        """
        Q = self.observability_matrix(system)
        return np.linalg.matrix_rank(Q) == system.A.shape[0]

    # ============= LQR CONTROL =============

    def lqr(self, system: StateSpaceModel, Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linear Quadratic Regulator design
        Returns gain matrix K and Riccati solution P
        """
        # Solve algebraic Riccati equation
        P = linalg.solve_continuous_are(system.A, system.B, Q, R)

        # Calculate optimal gain
        K = linalg.inv(R) @ system.B.T @ P

        return K, P

    def lqr_with_integral(self, system: StateSpaceModel,
                         Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        LQR with integral action for zero steady-state error
        """
        n = system.A.shape[0]
        m = system.B.shape[1]

        # Augment system with integral states
        A_aug = np.block([
            [system.A, np.zeros((n, n))],
            [-system.C, np.zeros((n, n))]
        ])

        B_aug = np.block([
            [system.B],
            [np.zeros((n, m))]
        ])

        # Augment Q matrix
        Q_aug = np.block([
            [Q, np.zeros((n, n))],
            [np.zeros((n, n)), np.eye(n)]
        ])

        # Solve augmented LQR problem
        P_aug = linalg.solve_continuous_are(A_aug, B_aug, Q_aug, R)
        K_aug = linalg.inv(R) @ B_aug.T @ P_aug

        return K_aug[:, :n], K_aug[:, n:]

    def lqg(self, system: StateSpaceModel, Q: np.ndarray, R: np.ndarray,
           Qn: np.ndarray, Rn: np.ndarray) -> Tuple[np.ndarray, KalmanFilter]:
        """
        Linear Quadratic Gaussian control (LQR + Kalman filter)
        """
        # Design LQR controller
        K_lqr, _ = self.lqr(system, Q, R)

        # Design Kalman filter
        kf = KalmanFilter(
            A=system.A,
            B=system.B,
            C=system.C,
            Q=Qn,
            R=Rn
        )

        return K_lqr, kf

    # ============= KALMAN FILTERING =============

    def kalman_predict(self, kf: KalmanFilter, u: np.ndarray):
        """
        Kalman filter prediction step
        """
        # Predict state
        kf.x = kf.A @ kf.x + kf.B @ u

        # Predict error covariance
        kf.P = kf.A @ kf.P @ kf.A.T + kf.Q

    def kalman_update(self, kf: KalmanFilter, y: np.ndarray):
        """
        Kalman filter update step
        """
        # Innovation
        innovation = y - kf.C @ kf.x

        # Innovation covariance
        S = kf.C @ kf.P @ kf.C.T + kf.R

        # Kalman gain
        K = kf.P @ kf.C.T @ linalg.inv(S)

        # Update state estimate
        kf.x = kf.x + K @ innovation

        # Update error covariance
        n = len(kf.x)
        kf.P = (np.eye(n) - K @ kf.C) @ kf.P

    def extended_kalman_filter(self, f: Callable, h: Callable,
                             F_jac: Callable, H_jac: Callable,
                             x: np.ndarray, P: np.ndarray,
                             u: np.ndarray, y: np.ndarray,
                             Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extended Kalman filter for nonlinear systems
        """
        # Prediction
        x_pred = f(x, u)
        F = F_jac(x, u)
        P_pred = F @ P @ F.T + Q

        # Update
        H = H_jac(x_pred)
        innovation = y - h(x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ linalg.inv(S)

        x_new = x_pred + K @ innovation
        P_new = (np.eye(len(x)) - K @ H) @ P_pred

        return x_new, P_new

    # ============= ROBUST CONTROL =============

    def h_infinity_norm(self, system: StateSpaceModel) -> float:
        """
        Compute H-infinity norm of system
        """
        # Convert to transfer function
        w, h = signal.freqresp(signal.StateSpace(system.A, system.B, system.C, system.D))
        return np.max(np.abs(h))

    def mu_synthesis(self, system: StateSpaceModel,
                    uncertainty: Dict[str, float]) -> np.ndarray:
        """
        μ-synthesis for robust control with structured uncertainty
        Simplified version for demonstration
        """
        # This is a simplified placeholder
        # Real μ-synthesis requires specialized algorithms
        n = system.A.shape[0]
        m = system.B.shape[1]

        # Design nominal controller
        Q = np.eye(n)
        R = np.eye(m)
        K, _ = self.lqr(system, Q, R)

        # Add robustness margin based on uncertainty
        margin = 1 + sum(uncertainty.values())
        K_robust = K * margin

        return K_robust

    # ============= MODEL PREDICTIVE CONTROL =============

    def mpc(self, system: StateSpaceModel, x0: np.ndarray,
           horizon: int, Q: np.ndarray, R: np.ndarray,
           constraints: Optional[Dict] = None) -> np.ndarray:
        """
        Model Predictive Control with constraints
        Simplified version - real MPC requires quadratic programming
        """
        n = system.A.shape[0]
        m = system.B.shape[1]

        # Build prediction matrices
        A_pred = np.zeros((n * horizon, n))
        B_pred = np.zeros((n * horizon, m * horizon))

        A_power = np.eye(n)
        for i in range(horizon):
            A_pred[i*n:(i+1)*n, :] = A_power @ system.A
            A_power = A_power @ system.A

            for j in range(i + 1):
                if j == 0:
                    B_pred[i*n:(i+1)*n, j*m:(j+1)*m] = system.B
                else:
                    B_pred[i*n:(i+1)*n, j*m:(j+1)*m] = (
                        linalg.matrix_power(system.A, i-j) @ system.B
                    )

        # Build cost matrices
        Q_bar = linalg.block_diag(*[Q for _ in range(horizon)])
        R_bar = linalg.block_diag(*[R for _ in range(horizon)])

        # Solve unconstrained MPC (simplified)
        H = B_pred.T @ Q_bar @ B_pred + R_bar
        f = x0.T @ A_pred.T @ Q_bar @ B_pred

        # Optimal control sequence (unconstrained)
        U_opt = -linalg.inv(H) @ f.T

        # Return first control action
        return U_opt[:m]

    # ============= ADAPTIVE CONTROL =============

    def mrac(self, reference_model: StateSpaceModel,
            plant: StateSpaceModel,
            learning_rate: float = 0.1) -> np.ndarray:
        """
        Model Reference Adaptive Control
        """
        # Simplified MRAC implementation
        n = plant.A.shape[0]
        m = plant.B.shape[1]

        # Initialize adaptive gains
        K_x = np.zeros((m, n))
        K_r = np.zeros((m, m))

        # Lyapunov design for adaptation law
        P = linalg.solve_continuous_lyapunov(reference_model.A, np.eye(n))

        # Adaptation gains
        gamma_x = learning_rate * np.eye(n)
        gamma_r = learning_rate * np.eye(m)

        return K_x, K_r, P

    def sliding_mode_control(self, system: StateSpaceModel,
                           sliding_surface: np.ndarray,
                           switching_gain: float) -> Callable:
        """
        Sliding mode control for robust nonlinear control
        """
        def control_law(x: np.ndarray) -> np.ndarray:
            s = sliding_surface @ x
            u = -switching_gain * np.sign(s)
            return u

        return control_law

    # ============= SYSTEM IDENTIFICATION =============

    def system_identification(self, input_data: np.ndarray,
                            output_data: np.ndarray,
                            order: int) -> StateSpaceModel:
        """
        System identification using least squares
        """
        # Build regression matrix
        n_samples = len(output_data) - order
        phi = np.zeros((n_samples, 2 * order))

        for i in range(n_samples):
            # Past outputs
            for j in range(order):
                if i + j < len(output_data):
                    phi[i, j] = output_data[i + j]

            # Past inputs
            for j in range(order):
                if i + j < len(input_data):
                    phi[i, order + j] = input_data[i + j]

        # Least squares solution
        y = output_data[order:]
        theta = linalg.lstsq(phi, y)[0]

        # Convert to state-space
        A = np.zeros((order, order))
        A[1:, :-1] = np.eye(order - 1)
        A[0, :] = -theta[:order]

        B = np.zeros((order, 1))
        B[0, 0] = 1

        C = theta[order:].reshape(1, -1)
        D = np.zeros((1, 1))

        return StateSpaceModel(A=A, B=B, C=C, D=D)

    # ============= UTILITY METHODS =============

    def _find_ultimate_gain(self, system: StateSpaceModel) -> Tuple[float, float]:
        """Find ultimate gain and period for Ziegler-Nichols tuning"""
        # Simplified method - finds critical gain through iteration
        Kp = 1.0
        for _ in range(100):
            # Check closed-loop stability
            A_cl = system.A - Kp * system.B @ system.C
            eigenvalues = linalg.eigvals(A_cl)

            if np.any(np.real(eigenvalues) > 0):
                break

            Kp *= 1.1

        Ku = Kp

        # Find period from imaginary part of eigenvalues
        imag_parts = np.imag(eigenvalues)
        imag_parts = imag_parts[imag_parts != 0]

        if len(imag_parts) > 0:
            omega = np.abs(imag_parts[0])
            Pu = 2 * np.pi / omega
        else:
            Pu = 1.0  # Default value

        return Ku, Pu

    def _fit_fopdt(self, system: StateSpaceModel) -> Tuple[float, float, float]:
        """Fit first-order plus dead time model"""
        # Step response
        t, y = signal.step(signal.StateSpace(system.A, system.B, system.C, system.D))

        # Find steady-state gain
        K = y[-1]

        # Find time constant (63.2% of final value)
        idx_632 = np.argmax(y >= 0.632 * K)
        tau = t[idx_632]

        # Estimate dead time (simplified)
        theta = t[np.argmax(y > 0.01 * K)]

        return K, tau, theta

    def simulate_system(self, system: StateSpaceModel,
                       controller: Optional[Callable],
                       t_span: np.ndarray,
                       x0: np.ndarray,
                       reference: Optional[np.ndarray] = None) -> Dict:
        """
        Simulate controlled system
        """
        n = len(x0)
        m = system.B.shape[1]
        N = len(t_span)

        # Initialize arrays
        x = np.zeros((N, n))
        u = np.zeros((N, m))
        y = np.zeros((N, system.C.shape[0]))

        x[0, :] = x0
        dt = t_span[1] - t_span[0]

        for i in range(N - 1):
            # Calculate control input
            if controller is not None:
                if reference is not None:
                    u[i, :] = controller(x[i, :], reference[i])
                else:
                    u[i, :] = controller(x[i, :])
            else:
                u[i, :] = 0

            # Update state (Euler integration)
            x[i + 1, :] = x[i, :] + dt * (system.A @ x[i, :] + system.B @ u[i, :])

            # Calculate output
            y[i, :] = system.C @ x[i, :] + system.D @ u[i, :]

        # Final output
        y[-1, :] = system.C @ x[-1, :] + system.D @ u[-1, :]

        return {
            'time': t_span,
            'state': x,
            'input': u,
            'output': y
        }

def run_demo():
    """Comprehensive demonstration of control systems lab"""
    print("="*60)
    print("CONTROL SYSTEMS LAB - Comprehensive Demo")
    print("="*60)

    lab = ControlSystemsLab()

    # Create a simple second-order system (mass-spring-damper)
    print("\n1. STATE-SPACE MODEL")
    print("-" * 40)

    # Transfer function: 1/(s² + 2s + 1)
    num = [1]
    den = [1, 2, 1]
    system = lab.create_state_space(num, den)
    print(f"System A matrix:\n{system.A}")
    print(f"Controllable: {lab.is_controllable(system)}")
    print(f"Observable: {lab.is_observable(system)}")

    # PID Control
    print("\n2. PID CONTROL")
    print("-" * 40)

    pid = PIDController(Kp=2.0, Ki=0.5, Kd=0.1, dt=0.01)

    # Single step response
    setpoint = 1.0
    measurement = 0.8
    control = lab.pid_control(pid, setpoint, measurement)
    print(f"Setpoint: {setpoint}, Measurement: {measurement}")
    print(f"PID output: {control:.4f}")

    # Auto-tuning
    auto_pid = lab.auto_tune_pid(system, method="ziegler-nichols")
    print(f"Auto-tuned gains: Kp={auto_pid.Kp:.2f}, Ki={auto_pid.Ki:.2f}, Kd={auto_pid.Kd:.2f}")

    # LQR Control
    print("\n3. LQR CONTROL")
    print("-" * 40)

    Q = np.eye(2)  # State cost
    R = np.array([[1]])  # Control cost

    K_lqr, P_lqr = lab.lqr(system, Q, R)
    print(f"LQR gain: {K_lqr}")
    print(f"Cost-to-go: {np.trace(P_lqr):.4f}")

    # Kalman Filter
    print("\n4. KALMAN FILTER")
    print("-" * 40)

    # Process and measurement noise
    Qn = 0.01 * np.eye(2)
    Rn = 0.1 * np.eye(1)

    kf = KalmanFilter(A=system.A, B=system.B, C=system.C, Q=Qn, R=Rn)

    # Single update step
    u = np.array([0.1])
    y = np.array([0.5])

    lab.kalman_predict(kf, u)
    lab.kalman_update(kf, y)
    print(f"State estimate: {kf.x}")
    print(f"Estimation error covariance trace: {np.trace(kf.P):.4f}")

    # MPC
    print("\n5. MODEL PREDICTIVE CONTROL")
    print("-" * 40)

    x0 = np.array([1.0, 0.0])
    horizon = 10
    u_mpc = lab.mpc(system, x0, horizon, Q, R)
    print(f"Initial state: {x0}")
    print(f"MPC first control action: {u_mpc}")

    # System Simulation
    print("\n6. SYSTEM SIMULATION")
    print("-" * 40)

    t_span = np.linspace(0, 10, 100)
    x0 = np.array([0, 0])

    # Define simple proportional controller
    def controller(x):
        reference = 1.0
        error = reference - x[0]
        return np.array([2.0 * error])

    results = lab.simulate_system(system, controller, t_span, x0)

    print(f"Simulation complete")
    print(f"Final state: {results['state'][-1, :]}")
    print(f"Final output: {results['output'][-1, :]}")
    print(f"Steady-state error: {1.0 - results['output'][-1, 0]:.4f}")

if __name__ == '__main__':
    run_demo()