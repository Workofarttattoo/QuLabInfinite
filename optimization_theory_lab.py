"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

OPTIMIZATION THEORY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.constants import pi

@dataclass
class OptimizationParameters:
    objective_function: callable
    initial_guess: np.ndarray = field(default_factory=lambda: np.zeros((1,), dtype=np.float64))
    bounds: tuple[tuple[float, float], ...] = field(default=None)
    constraints: list[dict[str, any]] = field(default=None)
    method: str = 'SLSQP'
    options: dict[str, any] = field(default_factory=lambda: {'disp': False})

@dataclass
class OptimizationResult:
    x: np.ndarray  # optimal solution
    fun: float     # objective function value at minimum
    success: bool  # whether the optimization terminated successfully
    message: str   # description of the cause of failure or termination

def optimize(parameters: OptimizationParameters) -> OptimizationResult:
    from scipy.optimize import minimize
    
    result = minimize(
        parameters.objective_function,
        parameters.initial_guess.astype(np.float64),
        method=parameters.method,
        bounds=np.array([b for b in parameters.bounds]) if parameters.bounds else None, 
        constraints=parameters.constraints,  
        options=parameters.options
    )
    
    return OptimizationResult(x=result.x, fun=result.fun, success=result.success, message=result.message)

def rosenbrock(x):
    """The Rosenbrock function is a non-convex function used as a performance test problem for optimization algorithms."""
    x = x.astype(np.float64)
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rastrigin(x):
    """The Rastrigin function is a non-convex function used as an optimization benchmark problem."""
    d = len(x)  # number of dimensions
    return 10*d + np.sum(x**2 - 10*np.cos(2*pi*x))

def run_demo():
    params_rosenbrock = OptimizationParameters(
        objective_function=rosenbrock,
        initial_guess=np.array([5, 5], dtype=np.float64),
        bounds=[(-10.0, 10.0)] * 2
    )
    
    result_rosenbrock = optimize(params_rosenbrock)
    print(f"Rosenbrock Function Optimal Solution: {result_rosenbrock.x}")
    print(f"Function Value at Minimum: {result_rosenbrock.fun}\n")
    
    params_rastrigin = OptimizationParameters(
        objective_function=rastrigin,
        initial_guess=np.array([5, 5], dtype=np.float64),
        bounds=[(-10.0, 10.0)] * 2
    )
    
    result_rastrigin = optimize(params_rastrigin)
    print(f"Rastrigin Function Optimal Solution: {result_rastrigin.x}")
    print(f"Function Value at Minimum: {result_rastrigin.fun}")

if __name__ == '__main__':
    run_demo()