"""
Design Optimizer Module
=======================

Multi-objective design optimization for virtual product development.
Supports parameter optimization, trade-off analysis, and design space exploration.

Copyright (c) Joshua Hendricks Cole (DBA: Corporation of Light)
PATENT PENDING - All Rights Reserved
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum, auto
import math
import random
from datetime import datetime

# Optional numpy import for advanced features
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


class OptimizationObjective(Enum):
    """Types of optimization objectives."""
    MINIMIZE = auto()
    MAXIMIZE = auto()
    TARGET = auto()  # Achieve a specific value


class OptimizationMethod(Enum):
    """Optimization algorithms available."""
    GRADIENT_DESCENT = auto()
    GENETIC_ALGORITHM = auto()
    SIMULATED_ANNEALING = auto()
    PARTICLE_SWARM = auto()
    BAYESIAN = auto()
    GRID_SEARCH = auto()
    MONTE_CARLO = auto()


@dataclass
class DesignVariable:
    """A design variable that can be optimized."""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float = 0.0  # 0 = continuous
    unit: str = ""
    description: str = ""
    discipline: str = ""  # mechanical, electrical, etc.
    locked: bool = False

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range."""
        if self.max_value == self.min_value:
            return 0.5
        return (value - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, norm_value: float) -> float:
        """Convert normalized value back to actual range."""
        value = self.min_value + norm_value * (self.max_value - self.min_value)
        if self.step_size > 0:
            value = round(value / self.step_size) * self.step_size
        return max(self.min_value, min(self.max_value, value))

    def random_value(self) -> float:
        """Generate a random value within bounds."""
        if self.step_size > 0:
            steps = int((self.max_value - self.min_value) / self.step_size)
            return self.min_value + random.randint(0, steps) * self.step_size
        return random.uniform(self.min_value, self.max_value)


@dataclass
class OptimizationConstraint:
    """A constraint on the design."""
    name: str
    expression: str  # Python expression as string
    limit_value: float
    constraint_type: str = "<=  # <=, >=, =="
    penalty_factor: float = 1000.0
    description: str = ""

    def evaluate(self, variables: Dict[str, float]) -> Tuple[float, bool]:
        """
        Evaluate constraint and return (violation_amount, is_satisfied).
        """
        try:
            value = eval(self.expression, {"__builtins__": {}}, variables)

            if self.constraint_type == "<=":
                violation = max(0, value - self.limit_value)
                satisfied = value <= self.limit_value
            elif self.constraint_type == ">=":
                violation = max(0, self.limit_value - value)
                satisfied = value >= self.limit_value
            else:  # ==
                violation = abs(value - self.limit_value)
                satisfied = violation < 1e-6

            return violation * self.penalty_factor, satisfied

        except Exception:
            return float('inf'), False


@dataclass
class OptimizationResult:
    """Results from an optimization run."""
    success: bool
    best_values: Dict[str, float]
    best_objectives: Dict[str, float]
    iterations: int
    convergence_history: List[Dict[str, float]]
    pareto_front: List[Dict[str, Any]]  # For multi-objective
    constraint_violations: Dict[str, float]
    computation_time_seconds: float
    method_used: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'success': self.success,
            'best_values': self.best_values,
            'best_objectives': self.best_objectives,
            'iterations': self.iterations,
            'convergence_history': self.convergence_history,
            'pareto_front': self.pareto_front,
            'constraint_violations': self.constraint_violations,
            'computation_time_seconds': self.computation_time_seconds,
            'method_used': self.method_used,
            'notes': self.notes
        }


@dataclass
class TradeOffStudy:
    """Results from a trade-off study between objectives."""
    name: str
    objective_x: str
    objective_y: str
    data_points: List[Dict[str, Any]]
    pareto_optimal_points: List[int]  # Indices of Pareto optimal points
    best_compromise: Optional[int] = None  # Index of best compromise

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'objective_x': self.objective_x,
            'objective_y': self.objective_y,
            'data_points': self.data_points,
            'pareto_optimal_points': self.pareto_optimal_points,
            'best_compromise': self.best_compromise
        }


class DesignOptimizer:
    """
    Multi-objective design optimization engine.

    Features:
    - Multiple optimization algorithms
    - Multi-objective optimization
    - Constraint handling
    - Trade-off studies
    - Sensitivity analysis
    - Design space exploration
    """

    def __init__(self):
        """Initialize the optimizer."""
        self.variables: Dict[str, DesignVariable] = {}
        self.objectives: Dict[str, Dict[str, Any]] = {}
        self.constraints: List[OptimizationConstraint] = []
        self.results_history: List[OptimizationResult] = []

        # Algorithm parameters
        self.max_iterations = 1000
        self.convergence_tolerance = 1e-6
        self.population_size = 50  # For GA, PSO
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

    def add_variable(self, variable: DesignVariable):
        """Add a design variable."""
        self.variables[variable.name] = variable

    def add_objective(self, name: str, expression: str,
                      objective_type: OptimizationObjective = OptimizationObjective.MINIMIZE,
                      target_value: Optional[float] = None,
                      weight: float = 1.0):
        """Add an optimization objective."""
        self.objectives[name] = {
            'expression': expression,
            'type': objective_type,
            'target': target_value,
            'weight': weight
        }

    def add_constraint(self, constraint: OptimizationConstraint):
        """Add a constraint."""
        self.constraints.append(constraint)

    def _evaluate_objectives(self, values: Dict[str, float]) -> Dict[str, float]:
        """Evaluate all objectives for given variable values."""
        results = {}
        for name, obj in self.objectives.items():
            try:
                value = eval(obj['expression'], {"__builtins__": {}, "math": math}, values)
                results[name] = value
            except Exception:
                results[name] = float('inf')
        return results

    def _calculate_fitness(self, values: Dict[str, float]) -> float:
        """Calculate overall fitness (lower is better)."""
        obj_values = self._evaluate_objectives(values)

        fitness = 0.0
        for name, obj in self.objectives.items():
            value = obj_values.get(name, float('inf'))
            weight = obj['weight']

            if obj['type'] == OptimizationObjective.MINIMIZE:
                fitness += weight * value
            elif obj['type'] == OptimizationObjective.MAXIMIZE:
                fitness -= weight * value
            else:  # TARGET
                target = obj['target'] or 0
                fitness += weight * abs(value - target)

        # Add constraint penalties
        for constraint in self.constraints:
            penalty, _ = constraint.evaluate(values)
            fitness += penalty

        return fitness

    def _random_solution(self) -> Dict[str, float]:
        """Generate a random solution."""
        return {
            name: var.random_value()
            for name, var in self.variables.items()
            if not var.locked
        }

    def optimize_genetic(self) -> OptimizationResult:
        """Optimize using genetic algorithm."""
        start_time = datetime.now()

        # Initialize population
        population = [self._random_solution() for _ in range(self.population_size)]
        fitness_scores = [self._calculate_fitness(ind) for ind in population]

        best_solution = population[fitness_scores.index(min(fitness_scores))]
        best_fitness = min(fitness_scores)
        convergence_history = []

        for iteration in range(self.max_iterations):
            # Selection (tournament)
            new_population = []
            for _ in range(self.population_size):
                i, j = random.sample(range(self.population_size), 2)
                winner = population[i] if fitness_scores[i] < fitness_scores[j] else population[j]
                new_population.append(winner.copy())

            # Crossover
            for i in range(0, self.population_size - 1, 2):
                if random.random() < self.crossover_rate:
                    parent1, parent2 = new_population[i], new_population[i + 1]
                    for var_name in self.variables:
                        if random.random() < 0.5:
                            parent1[var_name], parent2[var_name] = parent2[var_name], parent1[var_name]

            # Mutation
            for individual in new_population:
                for var_name, var in self.variables.items():
                    if not var.locked and random.random() < self.mutation_rate:
                        individual[var_name] = var.random_value()

            population = new_population
            fitness_scores = [self._calculate_fitness(ind) for ind in population]

            current_best_idx = fitness_scores.index(min(fitness_scores))
            if fitness_scores[current_best_idx] < best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_solution = population[current_best_idx].copy()

            convergence_history.append({
                'iteration': iteration,
                'best_fitness': best_fitness,
                'avg_fitness': sum(fitness_scores) / len(fitness_scores)
            })

            # Check convergence
            if len(convergence_history) > 50:
                recent = [h['best_fitness'] for h in convergence_history[-50:]]
                if max(recent) - min(recent) < self.convergence_tolerance:
                    break

        # Evaluate final solution
        best_objectives = self._evaluate_objectives(best_solution)
        constraint_violations = {}
        for constraint in self.constraints:
            violation, _ = constraint.evaluate(best_solution)
            constraint_violations[constraint.name] = violation

        elapsed = (datetime.now() - start_time).total_seconds()

        result = OptimizationResult(
            success=best_fitness < float('inf'),
            best_values=best_solution,
            best_objectives=best_objectives,
            iterations=len(convergence_history),
            convergence_history=convergence_history,
            pareto_front=[],
            constraint_violations=constraint_violations,
            computation_time_seconds=elapsed,
            method_used='Genetic Algorithm'
        )

        self.results_history.append(result)
        return result

    def optimize_simulated_annealing(self, initial_temp: float = 1000.0,
                                      cooling_rate: float = 0.995) -> OptimizationResult:
        """Optimize using simulated annealing."""
        start_time = datetime.now()

        current = self._random_solution()
        current_fitness = self._calculate_fitness(current)
        best = current.copy()
        best_fitness = current_fitness

        temperature = initial_temp
        convergence_history = []

        for iteration in range(self.max_iterations):
            # Generate neighbor
            neighbor = current.copy()
            var_name = random.choice(list(self.variables.keys()))
            var = self.variables[var_name]

            if not var.locked:
                delta = random.gauss(0, (var.max_value - var.min_value) * 0.1)
                neighbor[var_name] = max(var.min_value,
                                         min(var.max_value, current[var_name] + delta))

            neighbor_fitness = self._calculate_fitness(neighbor)

            # Accept or reject
            if neighbor_fitness < current_fitness:
                current = neighbor
                current_fitness = neighbor_fitness
            else:
                prob = math.exp(-(neighbor_fitness - current_fitness) / temperature)
                if random.random() < prob:
                    current = neighbor
                    current_fitness = neighbor_fitness

            if current_fitness < best_fitness:
                best = current.copy()
                best_fitness = current_fitness

            temperature *= cooling_rate

            convergence_history.append({
                'iteration': iteration,
                'best_fitness': best_fitness,
                'current_fitness': current_fitness,
                'temperature': temperature
            })

            if temperature < 0.001:
                break

        best_objectives = self._evaluate_objectives(best)
        constraint_violations = {}
        for constraint in self.constraints:
            violation, _ = constraint.evaluate(best)
            constraint_violations[constraint.name] = violation

        elapsed = (datetime.now() - start_time).total_seconds()

        result = OptimizationResult(
            success=best_fitness < float('inf'),
            best_values=best,
            best_objectives=best_objectives,
            iterations=len(convergence_history),
            convergence_history=convergence_history,
            pareto_front=[],
            constraint_violations=constraint_violations,
            computation_time_seconds=elapsed,
            method_used='Simulated Annealing'
        )

        self.results_history.append(result)
        return result

    def optimize(self, method: OptimizationMethod = OptimizationMethod.GENETIC_ALGORITHM) -> OptimizationResult:
        """Run optimization with specified method."""
        if method == OptimizationMethod.GENETIC_ALGORITHM:
            return self.optimize_genetic()
        elif method == OptimizationMethod.SIMULATED_ANNEALING:
            return self.optimize_simulated_annealing()
        else:
            return self.optimize_genetic()  # Default

    def sensitivity_analysis(self, base_solution: Dict[str, float],
                             perturbation: float = 0.05) -> Dict[str, Dict[str, float]]:
        """
        Perform sensitivity analysis around a solution.

        Returns sensitivity of each objective to each variable.
        """
        base_objectives = self._evaluate_objectives(base_solution)
        sensitivities = {}

        for var_name, var in self.variables.items():
            if var.locked:
                continue

            sensitivities[var_name] = {}
            delta = (var.max_value - var.min_value) * perturbation

            # Perturb positive
            perturbed = base_solution.copy()
            perturbed[var_name] = min(var.max_value, base_solution[var_name] + delta)
            obj_plus = self._evaluate_objectives(perturbed)

            # Perturb negative
            perturbed[var_name] = max(var.min_value, base_solution[var_name] - delta)
            obj_minus = self._evaluate_objectives(perturbed)

            # Calculate sensitivity (central difference)
            for obj_name in self.objectives:
                sensitivity = (obj_plus[obj_name] - obj_minus[obj_name]) / (2 * delta)
                sensitivities[var_name][obj_name] = sensitivity

        return sensitivities

    def trade_off_study(self, objective_x: str, objective_y: str,
                        num_points: int = 50) -> TradeOffStudy:
        """
        Generate trade-off curve between two objectives.
        """
        data_points = []

        for i in range(num_points):
            # Vary weight between objectives
            weight_x = i / (num_points - 1)
            weight_y = 1 - weight_x

            # Temporarily modify weights
            orig_weight_x = self.objectives[objective_x]['weight']
            orig_weight_y = self.objectives[objective_y]['weight']

            self.objectives[objective_x]['weight'] = weight_x
            self.objectives[objective_y]['weight'] = weight_y

            # Optimize
            result = self.optimize_genetic()

            # Store point
            data_points.append({
                'weight_x': weight_x,
                'weight_y': weight_y,
                objective_x: result.best_objectives[objective_x],
                objective_y: result.best_objectives[objective_y],
                'variables': result.best_values
            })

            # Restore weights
            self.objectives[objective_x]['weight'] = orig_weight_x
            self.objectives[objective_y]['weight'] = orig_weight_y

        # Find Pareto optimal points
        pareto_optimal = []
        for i, point in enumerate(data_points):
            is_dominated = False
            for j, other in enumerate(data_points):
                if i != j:
                    # Check if other dominates point
                    better_x = other[objective_x] <= point[objective_x]
                    better_y = other[objective_y] <= point[objective_y]
                    strictly_better = (other[objective_x] < point[objective_x] or
                                       other[objective_y] < point[objective_y])
                    if better_x and better_y and strictly_better:
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_optimal.append(i)

        # Find best compromise (closest to utopia point)
        if pareto_optimal:
            min_x = min(data_points[i][objective_x] for i in pareto_optimal)
            min_y = min(data_points[i][objective_y] for i in pareto_optimal)

            best_compromise = min(pareto_optimal, key=lambda i: (
                (data_points[i][objective_x] - min_x) ** 2 +
                (data_points[i][objective_y] - min_y) ** 2
            ))
        else:
            best_compromise = None

        return TradeOffStudy(
            name=f"{objective_x} vs {objective_y}",
            objective_x=objective_x,
            objective_y=objective_y,
            data_points=data_points,
            pareto_optimal_points=pareto_optimal,
            best_compromise=best_compromise
        )

    def design_space_exploration(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """
        Explore the design space using Latin Hypercube Sampling.
        """
        n_vars = len([v for v in self.variables.values() if not v.locked])
        samples = []

        # Generate Latin Hypercube samples
        for var_name, var in self.variables.items():
            if var.locked:
                continue

            # Create stratified samples
            if HAS_NUMPY:
                intervals = np.linspace(0, 1, num_samples + 1)
            else:
                # Fallback: manual linspace
                intervals = [i / num_samples for i in range(num_samples + 1)]
            points = []
            for i in range(num_samples):
                point = random.uniform(intervals[i], intervals[i + 1])
                points.append(var.denormalize(point))
            random.shuffle(points)

            if not samples:
                samples = [{var_name: p} for p in points]
            else:
                for i, p in enumerate(points):
                    samples[i][var_name] = p

        # Evaluate all samples
        results = []
        for sample in samples:
            objectives = self._evaluate_objectives(sample)
            fitness = self._calculate_fitness(sample)

            constraint_status = {}
            feasible = True
            for constraint in self.constraints:
                _, satisfied = constraint.evaluate(sample)
                constraint_status[constraint.name] = satisfied
                if not satisfied:
                    feasible = False

            results.append({
                'variables': sample,
                'objectives': objectives,
                'fitness': fitness,
                'feasible': feasible,
                'constraints': constraint_status
            })

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status and configuration."""
        return {
            'variables': {
                name: {
                    'current': var.current_value,
                    'min': var.min_value,
                    'max': var.max_value,
                    'locked': var.locked
                }
                for name, var in self.variables.items()
            },
            'objectives': self.objectives,
            'constraints': [c.name for c in self.constraints],
            'results_count': len(self.results_history),
            'parameters': {
                'max_iterations': self.max_iterations,
                'population_size': self.population_size,
                'mutation_rate': self.mutation_rate
            }
        }
