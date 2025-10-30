"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Orchestrator - Multi-Physics Experiment Coordination
DAG-based workflows with parallel execution and cross-validation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid

from .hive_mind_core import Task, TaskPriority, HiveMind, Agent
from .crystalline_intent import ParsedIntent, ExperimentDesign
from .temporal_bridge import TemporalBridge, TimeScale

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowNode:
    """Node in workflow DAG"""
    node_id: str
    node_type: str  # "task", "parallel", "sequential", "conditional"
    description: str
    task_spec: Optional[Dict[str, Any]] = None
    children: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowEdge:
    """Edge in workflow DAG"""
    source: str
    target: str
    edge_type: str = "sequential"  # "sequential", "parallel", "conditional"
    condition: Optional[Callable] = None


@dataclass
class MultiPhysicsExperiment:
    """Multi-department experiment specification"""
    experiment_id: str
    name: str
    description: str
    departments: List[str]  # ["physics", "chemistry", "materials", etc.]
    workflow: Dict[str, WorkflowNode]
    edges: List[WorkflowEdge]
    parameters: Dict[str, Any]
    expected_duration: float
    priority: TaskPriority = TaskPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """Execute DAG-based workflows with parallel execution"""

    def __init__(self, hive_mind: HiveMind):
        self.hive_mind = hive_mind
        self.workflows: Dict[str, MultiPhysicsExperiment] = {}
        self.workflow_status: Dict[str, WorkflowStatus] = {}
        self.node_results: Dict[str, Dict[str, Any]] = {}  # workflow_id -> {node_id -> result}

    def register_workflow(self, experiment: MultiPhysicsExperiment) -> str:
        """Register experiment workflow"""
        self.workflows[experiment.experiment_id] = experiment
        self.workflow_status[experiment.experiment_id] = WorkflowStatus.PENDING
        self.node_results[experiment.experiment_id] = {}
        LOG.info(f"[info] Registered workflow {experiment.experiment_id}: {experiment.name}")
        return experiment.experiment_id

    def _build_dependency_graph(self, workflow: MultiPhysicsExperiment) -> Dict[str, Set[str]]:
        """Build dependency graph from workflow"""
        graph = {node_id: set(node.dependencies) for node_id, node in workflow.workflow.items()}

        # Add edge dependencies
        for edge in workflow.edges:
            if edge.edge_type in ["sequential", "conditional"]:
                graph[edge.target].add(edge.source)

        return graph

    def _get_ready_nodes(self, workflow: MultiPhysicsExperiment,
                        completed: Set[str]) -> List[str]:
        """Get nodes ready for execution"""
        dep_graph = self._build_dependency_graph(workflow)
        ready = []

        for node_id, node in workflow.workflow.items():
            if node_id in completed:
                continue

            # Check if all dependencies are completed
            deps = dep_graph.get(node_id, set())
            if deps.issubset(completed):
                # Check conditional edges
                condition_met = True
                for edge in workflow.edges:
                    if edge.target == node_id and edge.condition:
                        source_result = self.node_results[workflow.experiment_id].get(edge.source)
                        if not edge.condition(source_result or {}):
                            condition_met = False
                            break

                if condition_met:
                    ready.append(node_id)

        return ready

    async def execute_workflow(self, experiment_id: str) -> Dict[str, Any]:
        """Execute workflow with parallel execution"""
        if experiment_id not in self.workflows:
            raise ValueError(f"Workflow {experiment_id} not found")

        workflow = self.workflows[experiment_id]
        self.workflow_status[experiment_id] = WorkflowStatus.RUNNING

        completed: Set[str] = set()
        failed: Set[str] = set()

        LOG.info(f"[info] Starting workflow {experiment_id}: {workflow.name}")

        while len(completed) + len(failed) < len(workflow.workflow):
            # Get nodes ready for execution
            ready_nodes = self._get_ready_nodes(workflow, completed)

            if not ready_nodes:
                if not failed:
                    # No ready nodes and no failures: deadlock or completion
                    LOG.warning(f"[warn] Workflow {experiment_id} deadlocked or completed")
                break

            # Execute ready nodes in parallel
            tasks = []
            for node_id in ready_nodes:
                node = workflow.workflow[node_id]
                tasks.append(self._execute_node(workflow, node))

            # Wait for all parallel tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for node_id, result in zip(ready_nodes, results):
                if isinstance(result, Exception):
                    LOG.error(f"[error] Node {node_id} failed: {result}")
                    failed.add(node_id)
                    self.node_results[experiment_id][node_id] = {"error": str(result)}
                else:
                    completed.add(node_id)
                    self.node_results[experiment_id][node_id] = result

            # Short delay between iterations
            await asyncio.sleep(0.1)

        # Determine final status
        if failed:
            self.workflow_status[experiment_id] = WorkflowStatus.FAILED
            status = "failed"
        else:
            self.workflow_status[experiment_id] = WorkflowStatus.COMPLETED
            status = "completed"

        LOG.info(f"[info] Workflow {experiment_id} {status}: {len(completed)} completed, {len(failed)} failed")

        return {
            "experiment_id": experiment_id,
            "status": status,
            "completed_nodes": len(completed),
            "failed_nodes": len(failed),
            "results": self.node_results[experiment_id]
        }

    async def _execute_node(self, workflow: MultiPhysicsExperiment,
                           node: WorkflowNode) -> Dict[str, Any]:
        """Execute single workflow node"""
        LOG.info(f"[info] Executing node {node.node_id}: {node.description}")

        if node.node_type == "task":
            # Create and submit task
            task_spec = node.task_spec or {}
            task = Task(
                task_id=str(uuid.uuid4()),
                task_type=task_spec.get("type", "generic"),
                description=node.description,
                priority=workflow.priority,
                required_capabilities=task_spec.get("capabilities", []),
                parameters=task_spec.get("parameters", {}),
                estimated_duration=task_spec.get("duration", 60.0)
            )

            # Submit to hive mind
            self.hive_mind.submit_task(task)

            # Wait for completion (simplified - would use callbacks in production)
            timeout = task_spec.get("timeout", 300.0)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if task.task_id in self.hive_mind.distributor.completed_tasks:
                    completed_task = self.hive_mind.distributor.completed_tasks[task.task_id]
                    if completed_task.error:
                        raise Exception(completed_task.error)
                    return completed_task.result
                await asyncio.sleep(0.5)

            raise TimeoutError(f"Node {node.node_id} timed out after {timeout}s")

        elif node.node_type == "parallel":
            # Execute children in parallel
            child_tasks = [self._execute_node(workflow, workflow.workflow[child_id])
                          for child_id in node.children]
            results = await asyncio.gather(*child_tasks)
            return {"parallel_results": results}

        elif node.node_type == "sequential":
            # Execute children sequentially
            results = []
            for child_id in node.children:
                result = await self._execute_node(workflow, workflow.workflow[child_id])
                results.append(result)
            return {"sequential_results": results}

        elif node.node_type == "conditional":
            # Execute based on condition
            if node.condition:
                prev_results = self.node_results[workflow.experiment_id]
                if node.condition(prev_results):
                    # Execute children
                    results = []
                    for child_id in node.children:
                        result = await self._execute_node(workflow, workflow.workflow[child_id])
                        results.append(result)
                    return {"conditional_results": results, "condition_met": True}
                else:
                    return {"condition_met": False}

        return {"node_type": node.node_type, "completed": True}


class ResultsValidator:
    """Cross-check and validate multi-department results"""

    def __init__(self):
        self.validation_rules: Dict[str, Callable] = {}

    def add_validation_rule(self, rule_name: str, rule_fn: Callable[[Dict[str, Any]], bool]) -> None:
        """Add validation rule"""
        self.validation_rules[rule_name] = rule_fn

    def cross_check_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-check results from multiple departments"""
        checks = {}

        # Energy conservation
        if "energy_initial" in results and "energy_final" in results:
            energy_conserved = abs(results["energy_final"] - results["energy_initial"]) < 0.01 * results["energy_initial"]
            checks["energy_conservation"] = energy_conserved

        # Mass conservation
        if "mass_initial" in results and "mass_final" in results:
            mass_conserved = abs(results["mass_final"] - results["mass_initial"]) < 0.01 * results["mass_initial"]
            checks["mass_conservation"] = mass_conserved

        # Consistency between departments
        if "physics_temperature" in results and "chemistry_temperature" in results:
            temp_consistent = abs(results["physics_temperature"] - results["chemistry_temperature"]) < 5.0
            checks["temperature_consistency"] = temp_consistent

        return checks

    def statistical_validation(self, results: List[Dict[str, Any]],
                              key: str, expected_mean: Optional[float] = None,
                              expected_std: Optional[float] = None) -> Dict[str, Any]:
        """Validate results statistically"""
        import numpy as np

        values = [r.get(key) for r in results if key in r and isinstance(r[key], (int, float))]

        if not values:
            return {"valid": False, "reason": "no_data"}

        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)

        validation = {
            "valid": True,
            "mean": float(mean),
            "std": float(std),
            "n": len(values)
        }

        # Check against expected values
        if expected_mean is not None:
            mean_error = abs(mean - expected_mean) / expected_mean
            validation["mean_error"] = float(mean_error)
            if mean_error > 0.1:  # >10% error
                validation["valid"] = False
                validation["reason"] = "mean_out_of_range"

        if expected_std is not None:
            std_error = abs(std - expected_std) / expected_std
            validation["std_error"] = float(std_error)
            if std_error > 0.2:  # >20% error
                validation["valid"] = False
                validation["reason"] = "std_out_of_range"

        return validation

    def detect_outliers(self, results: List[Dict[str, Any]], key: str,
                       threshold: float = 3.0) -> List[int]:
        """Detect outliers using z-score"""
        import numpy as np

        values = [r.get(key) for r in results if key in r and isinstance(r[key], (int, float))]

        if len(values) < 3:
            return []

        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return []

        z_scores = np.abs((values - mean) / std)
        outlier_indices = np.where(z_scores > threshold)[0].tolist()

        return outlier_indices

    def validate_workflow_results(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete workflow results"""
        validation = {
            "valid": True,
            "checks_passed": [],
            "checks_failed": [],
            "warnings": []
        }

        # Run all validation rules
        for rule_name, rule_fn in self.validation_rules.items():
            try:
                passed = rule_fn(workflow_results)
                if passed:
                    validation["checks_passed"].append(rule_name)
                else:
                    validation["checks_failed"].append(rule_name)
                    validation["valid"] = False
            except Exception as e:
                validation["warnings"].append(f"Rule {rule_name} failed: {e}")

        # Cross-checks
        cross_checks = self.cross_check_results(workflow_results)
        for check_name, passed in cross_checks.items():
            if passed:
                validation["checks_passed"].append(check_name)
            else:
                validation["checks_failed"].append(check_name)
                validation["valid"] = False

        return validation


class Orchestrator:
    """Main orchestrator for multi-physics experiments"""

    def __init__(self):
        self.hive_mind = HiveMind()
        self.workflow_engine = WorkflowEngine(self.hive_mind)
        self.validator = ResultsValidator()
        self.temporal_bridge = TemporalBridge()

    async def initialize(self) -> None:
        """Initialize orchestrator"""
        await self.hive_mind.start()
        LOG.info("[info] Orchestrator initialized")

    async def shutdown(self) -> None:
        """Shutdown orchestrator"""
        await self.hive_mind.stop()
        LOG.info("[info] Orchestrator shutdown")

    def create_aerogel_experiment(self) -> MultiPhysicsExperiment:
        """Example: Create multi-department aerogel experiment"""
        experiment_id = f"aerogel_{int(time.time())}"

        # Define workflow nodes
        nodes = {
            "load_material": WorkflowNode(
                node_id="load_material",
                node_type="task",
                description="Load aerogel material properties",
                task_spec={
                    "type": "materials_db_query",
                    "capabilities": ["materials_db"],
                    "parameters": {"material": "Airloy_X103"},
                    "duration": 5.0
                }
            ),
            "setup_environment": WorkflowNode(
                node_id="setup_environment",
                node_type="task",
                description="Set environmental conditions",
                task_spec={
                    "type": "environment_setup",
                    "capabilities": ["temperature_control", "pressure_control"],
                    "parameters": {"temperature": -200, "pressure": 0.001},
                    "duration": 10.0
                }
            ),
            "calculate_thermal": WorkflowNode(
                node_id="calculate_thermal",
                node_type="task",
                description="Calculate thermal conductivity",
                task_spec={
                    "type": "thermal_analysis",
                    "capabilities": ["thermal_test"],
                    "parameters": {"method": "steady_state"},
                    "duration": 60.0
                },
                dependencies=["load_material", "setup_environment"]
            ),
            "apply_wind": WorkflowNode(
                node_id="apply_wind",
                node_type="task",
                description="Apply wind load",
                task_spec={
                    "type": "fluid_dynamics",
                    "capabilities": ["fluid_dynamics"],
                    "parameters": {"wind_speed": 30, "direction": "parallel"},
                    "duration": 120.0
                },
                dependencies=["setup_environment"]
            ),
            "stress_analysis": WorkflowNode(
                node_id="stress_analysis",
                node_type="task",
                description="Calculate stress distribution",
                task_spec={
                    "type": "mechanical_analysis",
                    "capabilities": ["tensile_test"],
                    "parameters": {"load_type": "wind"},
                    "duration": 90.0
                },
                dependencies=["apply_wind", "load_material"]
            ),
            "phase_check": WorkflowNode(
                node_id="phase_check",
                node_type="task",
                description="Check for phase transitions",
                task_spec={
                    "type": "phase_analysis",
                    "capabilities": ["reaction_simulation"],
                    "parameters": {},
                    "duration": 30.0
                },
                dependencies=["setup_environment", "calculate_thermal"]
            ),
            "aggregate_results": WorkflowNode(
                node_id="aggregate_results",
                node_type="task",
                description="Aggregate all results",
                task_spec={
                    "type": "aggregation",
                    "capabilities": ["data_validation"],
                    "parameters": {},
                    "duration": 15.0
                },
                dependencies=["calculate_thermal", "stress_analysis", "phase_check"]
            )
        }

        # Define edges
        edges = []
        for node in nodes.values():
            for dep in node.dependencies:
                edges.append(WorkflowEdge(source=dep, target=node.node_id))

        experiment = MultiPhysicsExperiment(
            experiment_id=experiment_id,
            name="Aerogel Under Extreme Conditions",
            description="Multi-department test of aerogel at -200°C, 0.001 bar, 30 mph wind",
            departments=["materials", "environment", "physics", "chemistry"],
            workflow=nodes,
            edges=edges,
            parameters={},
            expected_duration=330.0,
            priority=TaskPriority.HIGH
        )

        return experiment

    async def execute_experiment(self, experiment: MultiPhysicsExperiment) -> Dict[str, Any]:
        """Execute multi-physics experiment"""
        # Register workflow
        self.workflow_engine.register_workflow(experiment)

        # Execute
        LOG.info(f"[info] Starting experiment: {experiment.name}")
        results = await self.workflow_engine.execute_workflow(experiment.experiment_id)

        # Validate
        validation = self.validator.validate_workflow_results(results["results"])
        results["validation"] = validation

        LOG.info(f"[info] Experiment completed: {experiment.name}, valid={validation['valid']}")

        return results


if __name__ == "__main__":
    async def demo():
        orchestrator = Orchestrator()
        await orchestrator.initialize()

        # Register agents
        from .hive_mind_core import create_standard_agents
        for agent in create_standard_agents():
            orchestrator.hive_mind.register_agent(agent)

        # Create aerogel experiment
        experiment = orchestrator.create_aerogel_experiment()

        # Execute (would run actual simulation in production)
        # results = await orchestrator.execute_experiment(experiment)
        # print(json.dumps(results, indent=2))

        print(f"Aerogel experiment created: {experiment.experiment_id}")
        print(f"  Nodes: {len(experiment.workflow)}")
        print(f"  Edges: {len(experiment.edges)}")
        print(f"  Expected duration: {experiment.expected_duration}s")

        await orchestrator.shutdown()

    asyncio.run(demo())
