"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Crystalline Intent - Goal Decomposition and Experiment Planning
NLP-based intent parsing, Design of Experiments (DOE), resource estimation
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class ExperimentType(Enum):
    """Types of experimental designs"""
    FULL_FACTORIAL = "full_factorial"
    FRACTIONAL_FACTORIAL = "fractional_factorial"
    RESPONSE_SURFACE = "response_surface"
    TAGUCHI = "taguchi"
    LATIN_HYPERCUBE = "latin_hypercube"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    ONE_FACTOR = "one_factor"
    SCREENING = "screening"


class OptimizationGoal(Enum):
    """Optimization objectives"""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"
    CONSTRAIN = "constrain"


@dataclass
class Parameter:
    """Experimental parameter specification"""
    name: str
    param_type: str  # "continuous", "discrete", "categorical"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    discrete_values: Optional[List[Any]] = None
    default_value: Any = None
    unit: Optional[str] = None


@dataclass
class Objective:
    """Optimization objective"""
    name: str
    goal: OptimizationGoal
    target_value: Optional[float] = None
    weight: float = 1.0  # For multi-objective optimization
    tolerance: Optional[float] = None


@dataclass
class Constraint:
    """Constraint on experiments"""
    name: str
    constraint_type: str  # "parameter", "resource", "safety", "feasibility"
    expression: str  # e.g., "temperature < 500"
    severity: str = "hard"  # "hard" or "soft"


@dataclass
class ExperimentDesign:
    """Complete experiment design specification"""
    design_id: str
    design_type: ExperimentType
    parameters: List[Parameter]
    objectives: List[Objective]
    constraints: List[Constraint]
    num_runs: int
    run_matrix: np.ndarray  # Design matrix
    estimated_duration: float  # Total time in seconds
    estimated_cost: float  # Computational cost
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedIntent:
    """Parsed experimental intent"""
    raw_query: str
    experiment_type: str
    materials: List[str]
    properties: List[str]
    conditions: Dict[str, Any]
    objectives: List[Objective]
    constraints: List[Constraint]
    confidence: float  # Parse confidence 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntentParser:
    """Parse natural language queries into experiment specifications"""

    def __init__(self):
        # Keywords for different intent categories
        self.experiment_keywords = {
            "test": ["test", "testing", "measure", "evaluate", "assess"],
            "optimize": ["optimize", "best", "improve", "enhance", "maximize", "minimize"],
            "compare": ["compare", "versus", "vs", "contrast", "difference"],
            "characterize": ["characterize", "analyze", "investigate", "study", "examine"],
            "synthesize": ["synthesize", "create", "produce", "generate", "make"],
            "predict": ["predict", "estimate", "forecast", "model", "simulate"]
        }

        self.property_keywords = {
            "strength": ["strength", "strong", "tough", "durable"],
            "conductivity": ["conductivity", "conductive", "resistivity"],
            "corrosion": ["corrosion", "rust", "oxidation", "degradation"],
            "thermal": ["thermal", "temperature", "heat", "cooling"],
            "optical": ["optical", "light", "absorption", "reflectance", "transparent"],
            "magnetic": ["magnetic", "magnetism", "ferromagnetic"],
            "elastic": ["elastic", "elasticity", "modulus", "stiffness"],
            "weight": ["weight", "lightweight", "heavy", "density", "mass"]
        }

        self.optimization_keywords = {
            OptimizationGoal.MAXIMIZE: ["maximize", "highest", "maximum", "increase", "best"],
            OptimizationGoal.MINIMIZE: ["minimize", "lowest", "minimum", "decrease", "reduce"],
            OptimizationGoal.TARGET: ["target", "achieve", "reach", "specific", "exactly"]
        }

    def parse(self, query: str) -> ParsedIntent:
        """Parse natural language query into structured intent"""
        query_lower = query.lower()

        # Identify experiment type
        exp_type = self._identify_experiment_type(query_lower)

        # Extract materials
        materials = self._extract_materials(query)

        # Extract properties
        properties = self._extract_properties(query_lower)

        # Extract conditions
        conditions = self._extract_conditions(query)

        # Extract objectives
        objectives = self._extract_objectives(query_lower, properties)

        # Extract constraints
        constraints = self._extract_constraints(query)

        # Compute confidence
        confidence = self._compute_confidence(exp_type, materials, properties, objectives)

        LOG.info(f"[info] Parsed intent: {exp_type}, {len(materials)} materials, {len(properties)} properties")

        return ParsedIntent(
            raw_query=query,
            experiment_type=exp_type,
            materials=materials,
            properties=properties,
            conditions=conditions,
            objectives=objectives,
            constraints=constraints,
            confidence=confidence
        )

    def _identify_experiment_type(self, query: str) -> str:
        """Identify type of experiment from query"""
        for exp_type, keywords in self.experiment_keywords.items():
            if any(kw in query for kw in keywords):
                return exp_type
        return "test"  # Default

    def _extract_materials(self, query: str) -> List[str]:
        """Extract material names from query"""
        # Common material patterns
        materials = []
        material_patterns = [
            r'\b(steel|aluminum|titanium|carbon fiber|graphene|silicon|copper|gold|silver)\b',
            r'\b(polymer|ceramic|composite|alloy|metal|glass|plastic)\b',
            r'\b([A-Z]+\s*\d+)\b'  # e.g., "AISI 304", "Al 6061"
        ]

        for pattern in material_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            materials.extend(matches)

        return list(set(materials))  # Remove duplicates

    def _extract_properties(self, query: str) -> List[str]:
        """Extract properties of interest from query"""
        properties = []
        for prop, keywords in self.property_keywords.items():
            if any(kw in query for kw in keywords):
                properties.append(prop)
        return properties

    def _extract_conditions(self, query: str) -> Dict[str, Any]:
        """Extract experimental conditions"""
        conditions = {}

        # Temperature
        temp_pattern = r'(-?\d+\.?\d*)\s*(°C|C|celsius|K|kelvin)'
        temp_matches = re.findall(temp_pattern, query, re.IGNORECASE)
        if temp_matches:
            conditions["temperature"] = float(temp_matches[0][0])
            conditions["temperature_unit"] = temp_matches[0][1]

        # Pressure
        pressure_pattern = r'(\d+\.?\d*)\s*(bar|Pa|psi|atm)'
        pressure_matches = re.findall(pressure_pattern, query, re.IGNORECASE)
        if pressure_matches:
            conditions["pressure"] = float(pressure_matches[0][0])
            conditions["pressure_unit"] = pressure_matches[0][1]

        # Time
        time_pattern = r'(\d+\.?\d*)\s*(seconds?|minutes?|hours?|days?|s|min|h|d)'
        time_matches = re.findall(time_pattern, query, re.IGNORECASE)
        if time_matches:
            conditions["duration"] = float(time_matches[0][0])
            conditions["duration_unit"] = time_matches[0][1]

        return conditions

    def _extract_objectives(self, query: str, properties: List[str]) -> List[Objective]:
        """Extract optimization objectives"""
        objectives = []

        for goal_type, keywords in self.optimization_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    # Find which property to optimize
                    for prop in properties:
                        if prop in query:
                            objectives.append(Objective(
                                name=prop,
                                goal=goal_type,
                                weight=1.0
                            ))

        # If no explicit optimization, assume characterization
        if not objectives and properties:
            for prop in properties:
                objectives.append(Objective(
                    name=prop,
                    goal=OptimizationGoal.MAXIMIZE,  # Default
                    weight=1.0
                ))

        return objectives

    def _extract_constraints(self, query: str) -> List[Constraint]:
        """Extract constraints from query"""
        constraints = []

        # Safety constraints
        if any(word in query.lower() for word in ["safe", "safety", "hazard"]):
            constraints.append(Constraint(
                name="safety",
                constraint_type="safety",
                expression="all_parameters_within_safe_limits",
                severity="hard"
            ))

        # Cost constraints
        if any(word in query.lower() for word in ["cheap", "affordable", "budget", "cost"]):
            constraints.append(Constraint(
                name="cost",
                constraint_type="resource",
                expression="cost < budget",
                severity="soft"
            ))

        return constraints

    def _compute_confidence(self, exp_type: str, materials: List[str],
                          properties: List[str], objectives: List[Objective]) -> float:
        """Compute confidence in parse"""
        score = 0.5  # Base confidence

        if exp_type != "test":
            score += 0.1
        if materials:
            score += 0.2
        if properties:
            score += 0.1
        if objectives:
            score += 0.1

        return min(score, 1.0)


class ExperimentDesigner:
    """Generate optimal experiment designs (DOE)"""

    def __init__(self):
        pass

    def design_full_factorial(self, parameters: List[Parameter], levels: int = 2) -> np.ndarray:
        """Generate full factorial design"""
        n_params = len(parameters)
        n_runs = levels ** n_params

        # Generate all combinations
        design = np.zeros((n_runs, n_params))
        for i in range(n_params):
            pattern = levels ** (n_params - i - 1)
            repetitions = levels ** i
            design[:, i] = np.tile(np.repeat(np.arange(levels), pattern), repetitions)

        # Scale to parameter ranges
        for i, param in enumerate(parameters):
            if param.param_type == "continuous":
                min_val = param.min_value if param.min_value is not None else 0.0
                max_val = param.max_value if param.max_value is not None else 1.0
                design[:, i] = min_val + design[:, i] * (max_val - min_val) / (levels - 1)

        return design

    def design_latin_hypercube(self, parameters: List[Parameter], num_samples: int) -> np.ndarray:
        """Generate Latin Hypercube Sampling design"""
        n_params = len(parameters)
        design = np.zeros((num_samples, n_params))

        for i in range(n_params):
            # Generate LHS samples in [0, 1]
            points = (np.random.permutation(num_samples) + np.random.rand(num_samples)) / num_samples

            # Scale to parameter range
            param = parameters[i]
            if param.param_type == "continuous":
                min_val = param.min_value if param.min_value is not None else 0.0
                max_val = param.max_value if param.max_value is not None else 1.0
                design[:, i] = min_val + points * (max_val - min_val)
            else:
                design[:, i] = points

        return design

    def design_response_surface(self, parameters: List[Parameter]) -> np.ndarray:
        """Generate Central Composite Design for response surface methodology"""
        n_params = len(parameters)

        # Center point
        center = np.zeros((1, n_params))

        # Factorial points (corners of hypercube)
        factorial = self.design_full_factorial(parameters, levels=2)

        # Axial points (star points)
        alpha = np.sqrt(n_params)  # Rotatable design
        axial = np.zeros((2 * n_params, n_params))
        for i in range(n_params):
            axial[2*i, i] = alpha
            axial[2*i+1, i] = -alpha

        # Combine all points
        design = np.vstack([factorial, axial, center])

        # Scale to parameter ranges
        for i, param in enumerate(parameters):
            if param.param_type == "continuous":
                min_val = param.min_value if param.min_value is not None else 0.0
                max_val = param.max_value if param.max_value is not None else 1.0
                # Normalize from [-alpha, alpha] to [min, max]
                design[:, i] = min_val + (design[:, i] + alpha) * (max_val - min_val) / (2 * alpha)

        return design

    def create_design(self, intent: ParsedIntent, design_type: ExperimentType,
                     num_runs: Optional[int] = None) -> ExperimentDesign:
        """Create experiment design from parsed intent"""
        # Define parameters from intent
        parameters = self._intent_to_parameters(intent)

        # Generate design matrix
        if design_type == ExperimentType.FULL_FACTORIAL:
            design_matrix = self.design_full_factorial(parameters, levels=2)
        elif design_type == ExperimentType.LATIN_HYPERCUBE:
            n_runs = num_runs if num_runs else 50
            design_matrix = self.design_latin_hypercube(parameters, n_runs)
        elif design_type == ExperimentType.RESPONSE_SURFACE:
            design_matrix = self.design_response_surface(parameters)
        else:
            # Default: Latin hypercube
            n_runs = num_runs if num_runs else 50
            design_matrix = self.design_latin_hypercube(parameters, n_runs)

        # Estimate resources
        estimated_duration = self._estimate_duration(design_matrix, intent)
        estimated_cost = self._estimate_cost(design_matrix, intent)

        design_id = f"design_{int(time.time() * 1000)}"

        LOG.info(f"[info] Created {design_type.value} design: {len(design_matrix)} runs, est. {estimated_duration:.1f}s")

        return ExperimentDesign(
            design_id=design_id,
            design_type=design_type,
            parameters=parameters,
            objectives=intent.objectives,
            constraints=intent.constraints,
            num_runs=len(design_matrix),
            run_matrix=design_matrix,
            estimated_duration=estimated_duration,
            estimated_cost=estimated_cost,
            metadata={"intent": intent.raw_query}
        )

    def _intent_to_parameters(self, intent: ParsedIntent) -> List[Parameter]:
        """Convert intent to parameter specifications"""
        parameters = []

        # Temperature parameter
        if "temperature" in intent.conditions:
            temp_val = intent.conditions["temperature"]
            parameters.append(Parameter(
                name="temperature",
                param_type="continuous",
                min_value=temp_val - 50,
                max_value=temp_val + 50,
                default_value=temp_val,
                unit="celsius"
            ))
        else:
            # Default temperature range
            parameters.append(Parameter(
                name="temperature",
                param_type="continuous",
                min_value=-50,
                max_value=500,
                default_value=25,
                unit="celsius"
            ))

        # Pressure parameter
        if "pressure" in intent.conditions:
            pressure_val = intent.conditions["pressure"]
            parameters.append(Parameter(
                name="pressure",
                param_type="continuous",
                min_value=max(0.01, pressure_val * 0.5),
                max_value=pressure_val * 2.0,
                default_value=pressure_val,
                unit="bar"
            ))

        # Material composition (if multiple materials)
        if len(intent.materials) > 1:
            parameters.append(Parameter(
                name="material_ratio",
                param_type="continuous",
                min_value=0.0,
                max_value=1.0,
                default_value=0.5,
                unit="fraction"
            ))

        return parameters

    def _estimate_duration(self, design_matrix: np.ndarray, intent: ParsedIntent) -> float:
        """Estimate total experiment duration"""
        # Base time per run
        base_time = 60.0  # 60 seconds

        # Adjust for experiment type
        if intent.experiment_type == "test":
            time_per_run = base_time
        elif intent.experiment_type == "optimize":
            time_per_run = base_time * 2
        elif intent.experiment_type == "synthesize":
            time_per_run = base_time * 5
        else:
            time_per_run = base_time

        return len(design_matrix) * time_per_run

    def _estimate_cost(self, design_matrix: np.ndarray, intent: ParsedIntent) -> float:
        """Estimate computational cost (arbitrary units)"""
        # Cost per run
        cost_per_run = 1.0

        # Adjust for complexity
        num_properties = len(intent.properties)
        complexity_factor = 1.0 + num_properties * 0.2

        return len(design_matrix) * cost_per_run * complexity_factor


import time


class ResourceEstimator:
    """Estimate computational resources required for experiments"""

    def estimate(self, design: ExperimentDesign) -> Dict[str, Any]:
        """Estimate resources for experiment design"""
        num_runs = design.num_runs
        num_params = len(design.parameters)

        # CPU cores (assume parallel execution)
        max_parallel = min(num_runs, 16)  # Cap at 16 cores
        cpu_cores = max_parallel

        # RAM per run (MB)
        ram_per_run = 100 + num_params * 50  # Base + parameter overhead
        total_ram = ram_per_run * max_parallel

        # GPU (optional, for accelerated simulations)
        gpu_required = any(obj.name in ["quantum", "molecular_dynamics"] for obj in design.objectives)

        # Execution time
        eta_seconds = design.estimated_duration
        eta_hours = eta_seconds / 3600

        # Cost-benefit score
        benefit = len(design.objectives) * design.num_runs
        cost = design.estimated_cost
        cost_benefit = benefit / max(cost, 0.1)

        return {
            "cpu_cores": cpu_cores,
            "ram_mb": total_ram,
            "gpu_required": gpu_required,
            "eta_seconds": eta_seconds,
            "eta_hours": eta_hours,
            "estimated_cost": cost,
            "cost_benefit_ratio": cost_benefit,
            "parallelizable": True,
            "max_parallel_runs": max_parallel
        }


class SuccessCriteria:
    """Define validation metrics and success thresholds"""

    def __init__(self):
        self.criteria: List[Dict[str, Any]] = []

    def add_criterion(self, name: str, metric: str, threshold: float,
                     direction: str = "maximize") -> None:
        """Add success criterion"""
        self.criteria.append({
            "name": name,
            "metric": metric,
            "threshold": threshold,
            "direction": direction
        })

    def evaluate(self, results: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate if results meet success criteria"""
        passed = []
        failed = []

        for criterion in self.criteria:
            metric = criterion["metric"]
            if metric not in results:
                failed.append({**criterion, "reason": "metric_not_found"})
                continue

            value = results[metric]
            threshold = criterion["threshold"]
            direction = criterion["direction"]

            if direction == "maximize" and value >= threshold:
                passed.append({**criterion, "value": value})
            elif direction == "minimize" and value <= threshold:
                passed.append({**criterion, "value": value})
            elif direction == "target" and abs(value - threshold) < threshold * 0.1:
                passed.append({**criterion, "value": value})
            else:
                failed.append({**criterion, "value": value, "reason": "threshold_not_met"})

        success_rate = len(passed) / len(self.criteria) if self.criteria else 0.0

        return {
            "success": len(failed) == 0,
            "success_rate": success_rate,
            "passed": passed,
            "failed": failed
        }


class RiskAssessment:
    """Assess risks and identify potential failures"""

    def assess(self, design: ExperimentDesign) -> Dict[str, Any]:
        """Assess risks for experiment design"""
        risks = []

        # Check parameter ranges
        for param in design.parameters:
            if param.param_type == "continuous":
                if param.max_value and param.max_value > 1000:
                    risks.append({
                        "type": "parameter_range",
                        "severity": "medium",
                        "parameter": param.name,
                        "message": f"{param.name} exceeds typical range"
                    })

        # Check number of runs
        if design.num_runs > 1000:
            risks.append({
                "type": "computational_cost",
                "severity": "high",
                "message": f"Large number of runs ({design.num_runs}) may be expensive"
            })

        # Check constraint feasibility
        for constraint in design.constraints:
            if constraint.severity == "hard":
                risks.append({
                    "type": "constraint_violation",
                    "severity": "high",
                    "constraint": constraint.name,
                    "message": f"Hard constraint '{constraint.name}' must be satisfied"
                })

        # Overall risk level
        severity_scores = {"low": 1, "medium": 2, "high": 3}
        max_severity = max([severity_scores[r["severity"]] for r in risks]) if risks else 0
        risk_level = ["none", "low", "medium", "high"][max_severity]

        # Fallback plans
        fallbacks = []
        if design.design_type == ExperimentType.FULL_FACTORIAL:
            fallbacks.append("Switch to fractional factorial if too expensive")
        if design.num_runs > 500:
            fallbacks.append("Use Latin hypercube sampling for initial screening")

        return {
            "risk_level": risk_level,
            "risks": risks,
            "num_risks": len(risks),
            "fallback_plans": fallbacks,
            "recommended_action": "proceed" if risk_level in ["none", "low"] else "review"
        }


if __name__ == "__main__":
    # Demo
    parser = IntentParser()
    designer = ExperimentDesigner()
    estimator = ResourceEstimator()
    risk_assessor = RiskAssessment()

    # Parse intent
    query = "Find lightweight corrosion-resistant alloy with tensile strength > 500 MPa at 200°C"
    intent = parser.parse(query)

    print(f"Parsed Intent:")
    print(f"  Type: {intent.experiment_type}")
    print(f"  Materials: {intent.materials}")
    print(f"  Properties: {intent.properties}")
    print(f"  Objectives: {[o.name for o in intent.objectives]}")
    print(f"  Confidence: {intent.confidence:.2f}")

    # Create design
    design = designer.create_design(intent, ExperimentType.LATIN_HYPERCUBE, num_runs=100)

    print(f"\nExperiment Design:")
    print(f"  Design ID: {design.design_id}")
    print(f"  Type: {design.design_type.value}")
    print(f"  Runs: {design.num_runs}")
    print(f"  Parameters: {[p.name for p in design.parameters]}")

    # Estimate resources
    resources = estimator.estimate(design)
    print(f"\nResource Estimate:")
    print(f"  CPU cores: {resources['cpu_cores']}")
    print(f"  RAM: {resources['ram_mb']} MB")
    print(f"  ETA: {resources['eta_hours']:.2f} hours")
    print(f"  Cost-benefit: {resources['cost_benefit_ratio']:.2f}")

    # Risk assessment
    risks = risk_assessor.assess(design)
    print(f"\nRisk Assessment:")
    print(f"  Risk level: {risks['risk_level']}")
    print(f"  Number of risks: {risks['num_risks']}")
    print(f"  Recommended action: {risks['recommended_action']}")
