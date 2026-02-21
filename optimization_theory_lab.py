"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

OPTIMIZATION THEORY LAB - Production Ready Implementation
Gradient descent, simplex, genetic algorithms, simulated annealing, and advanced optimization
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable, Any
from scipy import optimize
from scipy.constants import pi
import time
import random

# Configuration constants
DEFAULT_TOLERANCE = 1e-6
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_POPULATION_SIZE = 100

@dataclass
class OptimizationResult:
    """Container for optimization results"""
    x: np.ndarray  # Optimal solution
    fun: float  # Function value at optimum
    nit: int  # Number of iterations
    success: bool  # Convergence status
    message: str  # Status message
    history: Optional[List[float]] = None  # Optimization history
    time_elapsed: float = 0.0

@dataclass
class Constraints:
    """Optimization constraints"""
    A_eq: Optional[np.ndarray] = None  # Equality constraints: A_eq @ x = b_eq
    b_eq: Optional[np.ndarray] = None
    A_ineq: Optional[np.ndarray] = None  # Inequality constraints: A_ineq @ x <= b_ineq
    b_ineq: Optional[np.ndarray] = None
    bounds: Optional[List[Tuple[float, float]]] = None  # Variable bounds
    nonlinear: Optional[List[Callable]] = None  # Nonlinear constraints

class OptimizationTheoryLab:
    """Comprehensive optimization laboratory with classical and modern algorithms"""

    def __init__(self):
        self.benchmark_functions = self._initialize_benchmark_functions()
        self.optimization_history = []

    # ============= GRADIENT-BASED METHODS =============

    def gradient_descent(self, f: Callable, grad_f: Callable,
                        x0: np.ndarray, learning_rate: float = 0.01,
                        max_iter: int = DEFAULT_MAX_ITERATIONS,
                        tol: float = DEFAULT_TOLERANCE) -> OptimizationResult:
        """
        Basic gradient descent optimization
        """
        x = x0.copy()
        history = [f(x)]
        start_time = time.time()

        for iteration in range(max_iter):
            # Compute gradient
            grad = grad_f(x)

            # Check convergence
            if np.linalg.norm(grad) < tol:
                return OptimizationResult(
                    x=x, fun=f(x), nit=iteration,
                    success=True, message="Converged",
                    history=history, time_elapsed=time.time() - start_time
                )

            # Update
            x = x - learning_rate * grad
            history.append(f(x))

        return OptimizationResult(
            x=x, fun=f(x), nit=max_iter,
            success=False, message="Maximum iterations reached",
            history=history, time_elapsed=time.time() - start_time
        )

    def momentum_gradient_descent(self, f: Callable, grad_f: Callable,
                                 x0: np.ndarray, learning_rate: float = 0.01,
                                 momentum: float = 0.9,
                                 max_iter: int = DEFAULT_MAX_ITERATIONS,
                                 tol: float = DEFAULT_TOLERANCE) -> OptimizationResult:
        """
        Gradient descent with momentum (accelerated)
        """
        x = x0.copy()
        velocity = np.zeros_like(x)
        history = [f(x)]
        start_time = time.time()

        for iteration in range(max_iter):
            grad = grad_f(x)

            if np.linalg.norm(grad) < tol:
                return OptimizationResult(
                    x=x, fun=f(x), nit=iteration,
                    success=True, message="Converged",
                    history=history, time_elapsed=time.time() - start_time
                )

            # Update velocity and position
            velocity = momentum * velocity - learning_rate * grad
            x = x + velocity
            history.append(f(x))

        return OptimizationResult(
            x=x, fun=f(x), nit=max_iter,
            success=False, message="Maximum iterations reached",
            history=history, time_elapsed=time.time() - start_time
        )

    def adam_optimizer(self, f: Callable, grad_f: Callable,
                      x0: np.ndarray, learning_rate: float = 0.001,
                      beta1: float = 0.9, beta2: float = 0.999,
                      epsilon: float = 1e-8,
                      max_iter: int = DEFAULT_MAX_ITERATIONS,
                      tol: float = DEFAULT_TOLERANCE) -> OptimizationResult:
        """
        Adam (Adaptive Moment Estimation) optimizer
        """
        x = x0.copy()
        m = np.zeros_like(x)  # First moment
        v = np.zeros_like(x)  # Second moment
        history = [f(x)]
        start_time = time.time()

        for t in range(1, max_iter + 1):
            grad = grad_f(x)

            if np.linalg.norm(grad) < tol:
                return OptimizationResult(
                    x=x, fun=f(x), nit=t,
                    success=True, message="Converged",
                    history=history, time_elapsed=time.time() - start_time
                )

            # Update moments
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2

            # Bias correction
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            # Update parameters
            x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
            history.append(f(x))

        return OptimizationResult(
            x=x, fun=f(x), nit=max_iter,
            success=False, message="Maximum iterations reached",
            history=history, time_elapsed=time.time() - start_time
        )

    def newton_method(self, f: Callable, grad_f: Callable,
                     hess_f: Callable, x0: np.ndarray,
                     max_iter: int = DEFAULT_MAX_ITERATIONS,
                     tol: float = DEFAULT_TOLERANCE) -> OptimizationResult:
        """
        Newton's method for optimization (requires Hessian)
        """
        x = x0.copy()
        history = [f(x)]
        start_time = time.time()

        for iteration in range(max_iter):
            grad = grad_f(x)

            if np.linalg.norm(grad) < tol:
                return OptimizationResult(
                    x=x, fun=f(x), nit=iteration,
                    success=True, message="Converged",
                    history=history, time_elapsed=time.time() - start_time
                )

            # Compute Hessian and solve Newton step
            H = hess_f(x)
            try:
                # Solve H * delta = -grad
                delta = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                # Hessian singular, use gradient descent step
                delta = -grad

            # Line search for step size
            alpha = self._backtracking_line_search(f, x, grad, delta)
            x = x + alpha * delta
            history.append(f(x))

        return OptimizationResult(
            x=x, fun=f(x), nit=max_iter,
            success=False, message="Maximum iterations reached",
            history=history, time_elapsed=time.time() - start_time
        )

    def bfgs(self, f: Callable, grad_f: Callable, x0: np.ndarray,
            max_iter: int = DEFAULT_MAX_ITERATIONS,
            tol: float = DEFAULT_TOLERANCE) -> OptimizationResult:
        """
        BFGS quasi-Newton method
        """
        n = len(x0)
        x = x0.copy()
        B = np.eye(n)  # Initial Hessian approximation
        history = [f(x)]
        start_time = time.time()

        for iteration in range(max_iter):
            grad = grad_f(x)

            if np.linalg.norm(grad) < tol:
                return OptimizationResult(
                    x=x, fun=f(x), nit=iteration,
                    success=True, message="Converged",
                    history=history, time_elapsed=time.time() - start_time
                )

            # Compute search direction
            p = -np.linalg.solve(B, grad)

            # Line search
            alpha = self._backtracking_line_search(f, x, grad, p)

            # Update position
            s = alpha * p
            x_new = x + s
            grad_new = grad_f(x_new)

            # BFGS update
            y = grad_new - grad
            rho = 1.0 / (y.T @ s)

            if rho > 0:
                I = np.eye(n)
                B = (I - rho * np.outer(s, y)) @ B @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

            x = x_new
            history.append(f(x))

        return OptimizationResult(
            x=x, fun=f(x), nit=max_iter,
            success=False, message="Maximum iterations reached",
            history=history, time_elapsed=time.time() - start_time
        )

    def conjugate_gradient(self, f: Callable, grad_f: Callable,
                          x0: np.ndarray,
                          max_iter: int = DEFAULT_MAX_ITERATIONS,
                          tol: float = DEFAULT_TOLERANCE) -> OptimizationResult:
        """
        Nonlinear conjugate gradient method (Fletcher-Reeves)
        """
        x = x0.copy()
        g = grad_f(x)
        d = -g
        history = [f(x)]
        start_time = time.time()

        for iteration in range(max_iter):
            if np.linalg.norm(g) < tol:
                return OptimizationResult(
                    x=x, fun=f(x), nit=iteration,
                    success=True, message="Converged",
                    history=history, time_elapsed=time.time() - start_time
                )

            # Line search along direction d
            alpha = self._backtracking_line_search(f, x, g, d)
            x = x + alpha * d

            # New gradient
            g_new = grad_f(x)

            # Fletcher-Reeves update
            beta = np.dot(g_new, g_new) / np.dot(g, g)
            d = -g_new + beta * d
            g = g_new

            history.append(f(x))

        return OptimizationResult(
            x=x, fun=f(x), nit=max_iter,
            success=False, message="Maximum iterations reached",
            history=history, time_elapsed=time.time() - start_time
        )

    # ============= LINEAR PROGRAMMING =============

    def simplex_method(self, c: np.ndarray, A_ub: np.ndarray,
                      b_ub: np.ndarray,
                      A_eq: Optional[np.ndarray] = None,
                      b_eq: Optional[np.ndarray] = None,
                      bounds: Optional[List[Tuple[float, float]]] = None) -> OptimizationResult:
        """
        Simplex method for linear programming
        Minimize: c^T x
        Subject to: A_ub x <= b_ub, A_eq x = b_eq, bounds on x
        """
        start_time = time.time()

        # Use scipy's implementation
        result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub,
                                 A_eq=A_eq, b_eq=b_eq,
                                 bounds=bounds, method='highs')

        return OptimizationResult(
            x=result.x if result.x is not None else np.array([]),
            fun=result.fun if result.fun is not None else np.inf,
            nit=result.nit if hasattr(result, 'nit') else 0,
            success=result.success,
            message=result.message,
            time_elapsed=time.time() - start_time
        )

    def interior_point_method(self, f: Callable, grad_f: Callable,
                            constraints: Constraints,
                            x0: np.ndarray,
                            barrier_param: float = 10.0,
                            max_iter: int = DEFAULT_MAX_ITERATIONS,
                            tol: float = DEFAULT_TOLERANCE) -> OptimizationResult:
        """
        Interior point method for constrained optimization
        """
        x = x0.copy()
        mu = barrier_param
        history = []
        start_time = time.time()

        for iteration in range(max_iter):
            # Barrier function
            def barrier_f(x):
                val = f(x)
                # Add log barrier for inequality constraints
                if constraints.A_ineq is not None:
                    slack = constraints.b_ineq - constraints.A_ineq @ x
                    if np.any(slack <= 0):
                        return np.inf
                    val -= mu * np.sum(np.log(slack))
                return val

            # Gradient of barrier function
            def barrier_grad(x):
                grad = grad_f(x)
                if constraints.A_ineq is not None:
                    slack = constraints.b_ineq - constraints.A_ineq @ x
                    grad += mu * constraints.A_ineq.T @ (1 / slack)
                return grad

            # Minimize barrier function
            grad = barrier_grad(x)

            if np.linalg.norm(grad) < tol:
                return OptimizationResult(
                    x=x, fun=f(x), nit=iteration,
                    success=True, message="Converged",
                    history=history, time_elapsed=time.time() - start_time
                )

            # Newton step
            x = x - 0.1 * grad  # Simplified: using gradient instead of full Newton

            # Decrease barrier parameter
            mu *= 0.9

            history.append(f(x))

        return OptimizationResult(
            x=x, fun=f(x), nit=max_iter,
            success=False, message="Maximum iterations reached",
            history=history, time_elapsed=time.time() - start_time
        )

    # ============= METAHEURISTICS =============

    def genetic_algorithm(self, f: Callable, bounds: List[Tuple[float, float]],
                         population_size: int = DEFAULT_POPULATION_SIZE,
                         generations: int = 100,
                         crossover_rate: float = 0.8,
                         mutation_rate: float = 0.1) -> OptimizationResult:
        """
        Genetic algorithm for global optimization
        """
        n_vars = len(bounds)
        start_time = time.time()
        history = []

        # Initialize population
        population = np.array([
            [np.random.uniform(low, high) for low, high in bounds]
            for _ in range(population_size)
        ])

        best_individual = None
        best_fitness = np.inf

        for generation in range(generations):
            # Evaluate fitness
            fitness = np.array([f(ind) for ind in population])
            history.append(np.min(fitness))

            # Update best
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_individual = population[gen_best_idx].copy()

            # Selection (tournament)
            selected = []
            for _ in range(population_size):
                i, j = np.random.choice(population_size, 2, replace=False)
                selected.append(population[i] if fitness[i] < fitness[j] else population[j])
            selected = np.array(selected)

            # Crossover
            offspring = []
            for i in range(0, population_size, 2):
                parent1, parent2 = selected[i], selected[i + 1] if i + 1 < population_size else selected[0]

                if np.random.random() < crossover_rate:
                    # Single-point crossover
                    point = np.random.randint(1, n_vars)
                    child1 = np.concatenate([parent1[:point], parent2[point:]])
                    child2 = np.concatenate([parent2[:point], parent1[point:]])
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                offspring.extend([child1, child2])

            offspring = np.array(offspring[:population_size])

            # Mutation
            for i in range(population_size):
                if np.random.random() < mutation_rate:
                    gene = np.random.randint(n_vars)
                    offspring[i, gene] = np.random.uniform(bounds[gene][0], bounds[gene][1])

            # Enforce bounds
            for i, (low, high) in enumerate(bounds):
                offspring[:, i] = np.clip(offspring[:, i], low, high)

            population = offspring

        return OptimizationResult(
            x=best_individual,
            fun=best_fitness,
            nit=generations,
            success=True,
            message="Completed all generations",
            history=history,
            time_elapsed=time.time() - start_time
        )

    def simulated_annealing(self, f: Callable, x0: np.ndarray,
                           bounds: Optional[List[Tuple[float, float]]] = None,
                           T0: float = 100.0,
                           cooling_rate: float = 0.95,
                           max_iter: int = DEFAULT_MAX_ITERATIONS) -> OptimizationResult:
        """
        Simulated annealing for global optimization
        """
        x = x0.copy()
        best_x = x.copy()
        best_f = f(x)
        current_f = best_f

        T = T0
        history = [best_f]
        start_time = time.time()

        for iteration in range(max_iter):
            # Generate neighbor
            neighbor = x + np.random.normal(0, T / T0, size=x.shape)

            # Apply bounds if specified
            if bounds:
                for i, (low, high) in enumerate(bounds):
                    neighbor[i] = np.clip(neighbor[i], low, high)

            # Evaluate neighbor
            neighbor_f = f(neighbor)

            # Accept or reject
            delta = neighbor_f - current_f
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                x = neighbor
                current_f = neighbor_f

                if current_f < best_f:
                    best_x = x.copy()
                    best_f = current_f

            # Cool down
            T *= cooling_rate
            history.append(best_f)

            # Check convergence
            if T < 1e-10:
                break

        return OptimizationResult(
            x=best_x,
            fun=best_f,
            nit=iteration,
            success=True,
            message="Temperature cooled to minimum",
            history=history,
            time_elapsed=time.time() - start_time
        )

    def particle_swarm(self, f: Callable, bounds: List[Tuple[float, float]],
                      n_particles: int = 30,
                      max_iter: int = DEFAULT_MAX_ITERATIONS,
                      w: float = 0.7,  # Inertia weight
                      c1: float = 1.5,  # Cognitive parameter
                      c2: float = 1.5) -> OptimizationResult:  # Social parameter
        """
        Particle Swarm Optimization (PSO)
        """
        n_vars = len(bounds)
        start_time = time.time()
        history = []

        # Initialize particles
        particles = np.array([
            [np.random.uniform(low, high) for low, high in bounds]
            for _ in range(n_particles)
        ])

        velocities = np.zeros_like(particles)

        # Initialize personal and global bests
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([f(p) for p in particles])

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        for iteration in range(max_iter):
            # Update velocities and positions
            for i in range(n_particles):
                r1, r2 = np.random.random(n_vars), np.random.random(n_vars)

                velocities[i] = (w * velocities[i] +
                               c1 * r1 * (personal_best_positions[i] - particles[i]) +
                               c2 * r2 * (global_best_position - particles[i]))

                particles[i] = particles[i] + velocities[i]

                # Enforce bounds
                for j, (low, high) in enumerate(bounds):
                    particles[i, j] = np.clip(particles[i, j], low, high)

                # Update personal best
                score = f(particles[i])
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i].copy()

                    # Update global best
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = particles[i].copy()

            history.append(global_best_score)

            # Adaptive inertia weight
            w *= 0.99

        return OptimizationResult(
            x=global_best_position,
            fun=global_best_score,
            nit=max_iter,
            success=True,
            message="Completed all iterations",
            history=history,
            time_elapsed=time.time() - start_time
        )

    def differential_evolution(self, f: Callable,
                             bounds: List[Tuple[float, float]],
                             population_size: int = 50,
                             F: float = 0.8,  # Mutation factor
                             CR: float = 0.7,  # Crossover probability
                             max_iter: int = DEFAULT_MAX_ITERATIONS) -> OptimizationResult:
        """
        Differential Evolution algorithm
        """
        n_vars = len(bounds)
        start_time = time.time()
        history = []

        # Initialize population
        population = np.array([
            [np.random.uniform(low, high) for low, high in bounds]
            for _ in range(population_size)
        ])

        fitness = np.array([f(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx].copy()
        best_fitness = fitness[best_idx]

        for iteration in range(max_iter):
            for i in range(population_size):
                # Mutation: DE/rand/1
                candidates = list(range(population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])

                # Crossover
                trial = np.copy(population[i])
                crossover_mask = np.random.random(n_vars) < CR
                if not crossover_mask.any():
                    crossover_mask[np.random.randint(n_vars)] = True
                trial[crossover_mask] = mutant[crossover_mask]

                # Enforce bounds
                for j, (low, high) in enumerate(bounds):
                    trial[j] = np.clip(trial[j], low, high)

                # Selection
                trial_fitness = f(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best = trial.copy()
                        best_fitness = trial_fitness

            history.append(best_fitness)

        return OptimizationResult(
            x=best,
            fun=best_fitness,
            nit=max_iter,
            success=True,
            message="Completed all iterations",
            history=history,
            time_elapsed=time.time() - start_time
        )

    # ============= CONSTRAINED OPTIMIZATION =============

    def penalty_method(self, f: Callable, constraints: Constraints,
                      x0: np.ndarray, penalty_param: float = 1.0,
                      max_iter: int = DEFAULT_MAX_ITERATIONS) -> OptimizationResult:
        """
        Penalty method for constrained optimization
        """
        x = x0.copy()
        mu = penalty_param
        history = []
        start_time = time.time()

        for iteration in range(max_iter):
            # Penalty function
            def penalized_f(x):
                val = f(x)

                # Equality constraints penalty
                if constraints.A_eq is not None:
                    violation = constraints.A_eq @ x - constraints.b_eq
                    val += mu * np.sum(violation**2)

                # Inequality constraints penalty
                if constraints.A_ineq is not None:
                    violation = np.maximum(0, constraints.A_ineq @ x - constraints.b_ineq)
                    val += mu * np.sum(violation**2)

                return val

            # Minimize penalized function (simplified gradient descent)
            for _ in range(100):
                # Numerical gradient
                grad = self._numerical_gradient(penalized_f, x)
                x = x - 0.01 * grad

            # Increase penalty parameter
            mu *= 10
            history.append(f(x))

            # Check constraint satisfaction
            if self._check_constraints(x, constraints):
                break

        return OptimizationResult(
            x=x,
            fun=f(x),
            nit=iteration,
            success=self._check_constraints(x, constraints),
            message="Converged" if self._check_constraints(x, constraints) else "Max iterations",
            history=history,
            time_elapsed=time.time() - start_time
        )

    def augmented_lagrangian(self, f: Callable, grad_f: Callable,
                           constraints: Constraints,
                           x0: np.ndarray,
                           max_iter: int = DEFAULT_MAX_ITERATIONS) -> OptimizationResult:
        """
        Augmented Lagrangian method for constrained optimization
        """
        n = len(x0)
        x = x0.copy()

        # Initialize Lagrange multipliers
        lambda_eq = np.zeros(len(constraints.b_eq)) if constraints.A_eq is not None else np.array([])
        lambda_ineq = np.zeros(len(constraints.b_ineq)) if constraints.A_ineq is not None else np.array([])

        rho = 1.0  # Penalty parameter
        history = []
        start_time = time.time()

        for iteration in range(max_iter):
            # Augmented Lagrangian function
            def aug_lag(x):
                val = f(x)

                # Equality constraints
                if constraints.A_eq is not None:
                    h = constraints.A_eq @ x - constraints.b_eq
                    val += np.dot(lambda_eq, h) + 0.5 * rho * np.sum(h**2)

                # Inequality constraints
                if constraints.A_ineq is not None:
                    g = constraints.A_ineq @ x - constraints.b_ineq
                    for i, gi in enumerate(g):
                        if gi > 0 or lambda_ineq[i] > 0:
                            val += lambda_ineq[i] * gi + 0.5 * rho * gi**2

                return val

            # Minimize augmented Lagrangian (simplified)
            for _ in range(50):
                grad = self._numerical_gradient(aug_lag, x)
                x = x - 0.01 * grad

            # Update multipliers
            if constraints.A_eq is not None:
                h = constraints.A_eq @ x - constraints.b_eq
                lambda_eq = lambda_eq + rho * h

            if constraints.A_ineq is not None:
                g = constraints.A_ineq @ x - constraints.b_ineq
                lambda_ineq = np.maximum(0, lambda_ineq + rho * g)

            # Update penalty parameter
            rho *= 1.5
            history.append(f(x))

            # Check convergence
            if self._check_constraints(x, constraints, tol=1e-4):
                break

        return OptimizationResult(
            x=x,
            fun=f(x),
            nit=iteration,
            success=self._check_constraints(x, constraints, tol=1e-4),
            message="Converged" if self._check_constraints(x, constraints) else "Max iterations",
            history=history,
            time_elapsed=time.time() - start_time
        )

    # ============= MULTI-OBJECTIVE OPTIMIZATION =============

    def pareto_front(self, objectives: List[Callable],
                    bounds: List[Tuple[float, float]],
                    n_points: int = 100) -> List[np.ndarray]:
        """
        Find Pareto front for multi-objective optimization
        """
        n_vars = len(bounds)
        n_obj = len(objectives)

        # Generate random solutions
        solutions = np.array([
            [np.random.uniform(low, high) for low, high in bounds]
            for _ in range(n_points * 10)
        ])

        # Evaluate objectives
        obj_values = np.array([
            [obj(sol) for obj in objectives]
            for sol in solutions
        ])

        # Find Pareto front
        pareto_front = []
        for i, obj_i in enumerate(obj_values):
            is_dominated = False
            for j, obj_j in enumerate(obj_values):
                if i != j and all(obj_j <= obj_i) and any(obj_j < obj_i):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(solutions[i])

        return pareto_front

    def weighted_sum_method(self, objectives: List[Callable],
                          weights: np.ndarray,
                          bounds: List[Tuple[float, float]],
                          **kwargs) -> OptimizationResult:
        """
        Weighted sum method for multi-objective optimization
        """
        def combined_objective(x):
            return sum(w * obj(x) for w, obj in zip(weights, objectives))

        # Use genetic algorithm for global search
        return self.genetic_algorithm(combined_objective, bounds, **kwargs)

    # ============= UTILITY METHODS =============

    def _backtracking_line_search(self, f: Callable, x: np.ndarray,
                                 grad: np.ndarray, direction: np.ndarray,
                                 alpha: float = 1.0, rho: float = 0.5,
                                 c: float = 1e-4) -> float:
        """
        Backtracking line search with Armijo condition
        """
        while f(x + alpha * direction) > f(x) + c * alpha * np.dot(grad, direction):
            alpha *= rho
            if alpha < 1e-10:
                break
        return alpha

    def _numerical_gradient(self, f: Callable, x: np.ndarray,
                          epsilon: float = 1e-8) -> np.ndarray:
        """
        Compute gradient numerically using finite differences
        """
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
        return grad

    def _numerical_hessian(self, f: Callable, x: np.ndarray,
                         epsilon: float = 1e-5) -> np.ndarray:
        """
        Compute Hessian numerically using finite differences
        """
        n = len(x)
        H = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()

                x_pp[i] += epsilon
                x_pp[j] += epsilon
                x_pm[i] += epsilon
                x_pm[j] -= epsilon
                x_mp[i] -= epsilon
                x_mp[j] += epsilon
                x_mm[i] -= epsilon
                x_mm[j] -= epsilon

                H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * epsilon**2)
                H[j, i] = H[i, j]

        return H

    def _check_constraints(self, x: np.ndarray, constraints: Constraints,
                         tol: float = 1e-6) -> bool:
        """
        Check if constraints are satisfied
        """
        # Equality constraints
        if constraints.A_eq is not None:
            if np.any(np.abs(constraints.A_eq @ x - constraints.b_eq) > tol):
                return False

        # Inequality constraints
        if constraints.A_ineq is not None:
            if np.any(constraints.A_ineq @ x - constraints.b_ineq > tol):
                return False

        # Bounds
        if constraints.bounds is not None:
            for i, (low, high) in enumerate(constraints.bounds):
                if x[i] < low - tol or x[i] > high + tol:
                    return False

        return True

    def _initialize_benchmark_functions(self) -> Dict[str, Callable]:
        """
        Initialize standard benchmark functions for testing
        """
        functions = {}

        # Rosenbrock function
        functions['rosenbrock'] = lambda x: sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                                               for i in range(len(x) - 1))

        # Rastrigin function
        functions['rastrigin'] = lambda x: 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * pi * xi)
                                                            for xi in x)

        # Sphere function
        functions['sphere'] = lambda x: sum(xi**2 for xi in x)

        # Ackley function
        functions['ackley'] = lambda x: (-20 * np.exp(-0.2 * np.sqrt(sum(xi**2 for xi in x) / len(x)))
                                        - np.exp(sum(np.cos(2 * pi * xi) for xi in x) / len(x))
                                        + 20 + np.e)

        # Griewank function
        functions['griewank'] = lambda x: (sum(xi**2 for xi in x) / 4000
                                          - np.prod(np.cos(x[i] / np.sqrt(i + 1)) for i in range(len(x)))
                                          + 1)

        return functions

    def benchmark_algorithms(self, function_name: str, dimension: int = 2,
                           algorithms: Optional[List[str]] = None) -> Dict[str, OptimizationResult]:
        """
        Benchmark multiple algorithms on a test function
        """
        if function_name not in self.benchmark_functions:
            raise ValueError(f"Unknown function: {function_name}")

        f = self.benchmark_functions[function_name]
        bounds = [(-10, 10)] * dimension
        x0 = np.random.uniform(-5, 5, dimension)

        if algorithms is None:
            algorithms = ['gradient_descent', 'adam', 'genetic_algorithm',
                         'simulated_annealing', 'particle_swarm']

        results = {}

        for algo_name in algorithms:
            if algo_name == 'gradient_descent':
                grad = lambda x: self._numerical_gradient(f, x)
                results[algo_name] = self.gradient_descent(f, grad, x0)
            elif algo_name == 'adam':
                grad = lambda x: self._numerical_gradient(f, x)
                results[algo_name] = self.adam_optimizer(f, grad, x0)
            elif algo_name == 'genetic_algorithm':
                results[algo_name] = self.genetic_algorithm(f, bounds, generations=50)
            elif algo_name == 'simulated_annealing':
                results[algo_name] = self.simulated_annealing(f, x0, bounds)
            elif algo_name == 'particle_swarm':
                results[algo_name] = self.particle_swarm(f, bounds, max_iter=100)

        return results

def run_demo():
    """Comprehensive demonstration of optimization theory lab"""
    print("="*60)
    print("OPTIMIZATION THEORY LAB - Comprehensive Demo")
    print("="*60)

    lab = OptimizationTheoryLab()

    # Test function: Rosenbrock
    print("\n1. GRADIENT-BASED OPTIMIZATION")
    print("-" * 40)

    f = lab.benchmark_functions['rosenbrock']
    grad_f = lambda x: lab._numerical_gradient(f, x)
    x0 = np.array([0, 0])

    # Gradient descent
    result = lab.gradient_descent(f, grad_f, x0, learning_rate=0.001, max_iter=1000)
    print(f"Gradient Descent: x = {result.x}, f(x) = {result.fun:.6f}, iterations = {result.nit}")

    # Adam optimizer
    result = lab.adam_optimizer(f, grad_f, x0, max_iter=1000)
    print(f"Adam: x = {result.x}, f(x) = {result.fun:.6f}, iterations = {result.nit}")

    # BFGS
    result = lab.bfgs(f, grad_f, x0, max_iter=100)
    print(f"BFGS: x = {result.x}, f(x) = {result.fun:.6f}, iterations = {result.nit}")

    # Linear programming
    print("\n2. LINEAR PROGRAMMING (SIMPLEX)")
    print("-" * 40)

    # Minimize: -x - 2y
    # Subject to: x + 2y <= 4, x >= 0, y >= 0
    c = np.array([-1, -2])
    A_ub = np.array([[1, 2]])
    b_ub = np.array([4])
    bounds = [(0, None), (0, None)]

    result = lab.simplex_method(c, A_ub, b_ub, bounds=bounds)
    print(f"Optimal solution: x = {result.x}")
    print(f"Optimal value: {result.fun:.4f}")

    # Metaheuristics
    print("\n3. METAHEURISTICS")
    print("-" * 40)

    # Test on Rastrigin function (many local minima)
    f = lab.benchmark_functions['rastrigin']
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]

    # Genetic Algorithm
    result = lab.genetic_algorithm(f, bounds, population_size=50, generations=100)
    print(f"Genetic Algorithm: x = {result.x}, f(x) = {result.fun:.6f}")

    # Simulated Annealing
    x0 = np.random.uniform(-5, 5, 2)
    result = lab.simulated_annealing(f, x0, bounds, max_iter=1000)
    print(f"Simulated Annealing: x = {result.x}, f(x) = {result.fun:.6f}")

    # Particle Swarm
    result = lab.particle_swarm(f, bounds, n_particles=30, max_iter=100)
    print(f"Particle Swarm: x = {result.x}, f(x) = {result.fun:.6f}")

    # Differential Evolution
    result = lab.differential_evolution(f, bounds, population_size=30, max_iter=100)
    print(f"Differential Evolution: x = {result.x}, f(x) = {result.fun:.6f}")

    # Constrained optimization
    print("\n4. CONSTRAINED OPTIMIZATION")
    print("-" * 40)

    # Minimize: x² + y²
    # Subject to: x + y >= 1
    f = lambda x: x[0]**2 + x[1]**2
    constraints = Constraints(
        A_ineq=np.array([[-1, -1]]),  # -x - y <= -1 => x + y >= 1
        b_ineq=np.array([-1])
    )
    x0 = np.array([2, 2])

    result = lab.penalty_method(f, constraints, x0, max_iter=50)
    print(f"Penalty Method: x = {result.x}, f(x) = {result.fun:.6f}")
    print(f"Constraint satisfied: {result.x[0] + result.x[1] >= 1}")

    # Benchmark comparison
    print("\n5. ALGORITHM BENCHMARK")
    print("-" * 40)

    results = lab.benchmark_algorithms('sphere', dimension=5,
                                      algorithms=['gradient_descent', 'adam',
                                                 'particle_swarm', 'genetic_algorithm'])

    print("Results on 5D Sphere Function:")
    for algo_name, result in results.items():
        print(f"  {algo_name}: f(x) = {result.fun:.6f}, time = {result.time_elapsed:.3f}s")

if __name__ == '__main__':
    run_demo()