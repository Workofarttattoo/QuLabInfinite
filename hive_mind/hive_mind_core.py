"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Hive Mind Core - Multi-Agent Coordination for QuLab Infinite
Level-6 Autonomous Agent Registry, Task Distribution, and Result Aggregation
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of specialized laboratory agents"""
    PHYSICS = "physics"
    QUANTUM = "quantum"
    MATERIALS = "materials"
    CHEMISTRY = "chemistry"
    ENVIRONMENT = "environment"
    VALIDATION = "validation"
    ORCHESTRATION = "orchestration"
    TEMPORAL = "temporal"


class TaskPriority(Enum):
    """Priority levels for task queue"""
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1


class TaskStatus(Enum):
    """Task lifecycle states"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Agent:
    """Represents a Level-6 autonomous laboratory agent"""
    agent_id: str
    agent_type: AgentType
    capabilities: List[str]
    current_load: float = 0.0  # 0.0 to 1.0
    max_concurrent_tasks: int = 5
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=lambda: {
        "tasks_completed": 0,
        "avg_completion_time": 0.0,
        "success_rate": 1.0,
        "quality_score": 1.0
    })

    def can_handle_task(self, task: 'Task') -> bool:
        """Check if agent can handle this task type"""
        if self.current_load >= 1.0:
            return False
        if len(task.required_capabilities) == 0:
            return True
        return any(cap in self.capabilities for cap in task.required_capabilities)

    def estimate_time(self, task: 'Task') -> float:
        """Estimate time to complete task (seconds)"""
        base_time = task.estimated_duration
        # Adjust for current load
        load_factor = 1.0 + self.current_load
        # Adjust for agent performance
        performance_factor = 2.0 - self.performance_metrics.get("quality_score", 1.0)
        return base_time * load_factor * performance_factor


@dataclass
class Task:
    """Represents an experiment or computation task"""
    task_id: str
    task_type: str
    description: str
    priority: TaskPriority
    required_capabilities: List[str]
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)  # task_ids that must complete first
    estimated_duration: float = 60.0  # seconds
    deadline: Optional[float] = None  # unix timestamp
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Aggregated result from multi-agent experiment"""
    experiment_id: str
    task_ids: List[str]
    agent_contributions: Dict[str, Any]
    merged_data: Dict[str, Any]
    validation_status: str
    confidence: float  # 0.0 to 1.0
    cross_validation_passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AgentRegistry:
    """Registry of all active laboratory agents"""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.agent_instances: Dict[str, 'Level6AgentInterface'] = {}
        self.agent_capabilities_index: Dict[str, Set[str]] = defaultdict(set)

    def register_agent(self, agent: Agent, agent_instance: 'Level6AgentInterface' = None) -> None:
        """Register a new agent in the hive mind"""
        self.agents[agent.agent_id] = agent
        if agent_instance:
            self.agent_instances[agent.agent_id] = agent_instance
        # Index capabilities for fast lookup
        for capability in agent.capabilities:
            self.agent_capabilities_index[capability].add(agent.agent_id)
        LOG.info(f"[info] Registered agent {agent.agent_id} ({agent.agent_type.value}) with {len(agent.capabilities)} capabilities")

    def unregister_agent(self, agent_id: str) -> None:
        """Remove agent from registry"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            # Remove from capability index
            for capability in agent.capabilities:
                self.agent_capabilities_index[capability].discard(agent_id)
            del self.agents[agent_id]
            if agent_id in self.agent_instances:
                del self.agent_instances[agent_id]
            LOG.info(f"[info] Unregistered agent {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def get_agent_instance(self, agent_id: str) -> Optional['Level6AgentInterface']:
        """Get agent instance by ID"""
        return self.agent_instances.get(agent_id)

    def find_agents_by_capability(self, capability: str) -> List[Agent]:
        """Find all agents with specific capability"""
        agent_ids = self.agent_capabilities_index.get(capability, set())
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]

    def find_agents_by_type(self, agent_type: AgentType) -> List[Agent]:
        """Find all agents of specific type"""
        return [agent for agent in self.agents.values() if agent.agent_type == agent_type]

    def get_available_agents(self) -> List[Agent]:
        """Get all agents with available capacity"""
        return [agent for agent in self.agents.values()
                if agent.status == "active" and agent.current_load < 1.0]

    def update_agent_load(self, agent_id: str, load_delta: float) -> None:
        """Update agent's current load"""
        if agent_id in self.agents:
            self.agents[agent_id].current_load = max(0.0, min(1.0,
                self.agents[agent_id].current_load + load_delta))

    def get_status_summary(self) -> Dict[str, Any]:
        """Get overview of all agents"""
        return {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a.status == "active"),
            "by_type": {
                agent_type.value: len(self.find_agents_by_type(agent_type))
                for agent_type in AgentType
            },
            "average_load": sum(a.current_load for a in self.agents.values()) / max(len(self.agents), 1),
            "capabilities": list(self.agent_capabilities_index.keys())
        }


class TaskDistributor:
    """Intelligent task distribution with load balancing"""

    def __init__(self, registry: AgentRegistry, broadcast_callback: Optional[Callable] = None):
        self.registry = registry
        self.broadcast_callback = broadcast_callback
        self.task_queue: Dict[TaskPriority, List[Task]] = {
            priority: [] for priority in TaskPriority
        }
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.task_dependency_graph: Dict[str, Set[str]] = defaultdict(set)

    def submit_task(self, task: Task) -> str:
        """Submit task to distribution queue"""
        # Add to priority queue
        self.task_queue[task.priority].append(task)
        # Build dependency graph
        for dep_id in task.dependencies:
            self.task_dependency_graph[task.task_id].add(dep_id)
        LOG.info(f"[info] Task {task.task_id} submitted: {task.description}")
        return task.task_id

    def submit_tasks(self, tasks: List[Task]) -> List[str]:
        """Submit multiple tasks"""
        return [self.submit_task(task) for task in tasks]

    def _is_task_ready(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        if not task.dependencies:
            return True
        return all(dep_id in self.completed_tasks and
                   self.completed_tasks[dep_id].status == TaskStatus.COMPLETED
                   for dep_id in task.dependencies)

    def _select_best_agent(self, task: Task) -> Optional[Agent]:
        """Select optimal agent for task using multi-criteria scoring"""
        available_agents = [agent for agent in self.registry.get_available_agents()
                           if agent.can_handle_task(task)]

        if not available_agents:
            return None

        # Score each agent
        scores = []
        for agent in available_agents:
            # Criteria: capability match, load, performance, estimated time
            capability_score = sum(1 for cap in task.required_capabilities
                                  if cap in agent.capabilities) / max(len(task.required_capabilities), 1)
            load_score = 1.0 - agent.current_load
            performance_score = agent.performance_metrics["quality_score"]
            time_estimate = agent.estimate_time(task)
            time_score = 1.0 / (1.0 + time_estimate / 60.0)  # Prefer faster agents

            # Weighted combination
            total_score = (0.4 * capability_score +
                          0.3 * load_score +
                          0.2 * performance_score +
                          0.1 * time_score)
            scores.append((agent, total_score))

        # Return best agent
        best_agent, best_score = max(scores, key=lambda x: x[1])
        return best_agent

    async def distribute_next(self) -> Optional[str]:
        """Distribute next task from queue to best available agent"""
        # Check queues in priority order
        for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
            queue = self.task_queue[priority]
            # Find first ready task
            for i, task in enumerate(queue):
                if self._is_task_ready(task):
                    # Find agent
                    agent = self._select_best_agent(task)
                    if agent:
                        # Remove from queue
                        queue.pop(i)
                        # Assign task
                        task.status = TaskStatus.ASSIGNED
                        task.assigned_agent = agent.agent_id
                        task.started_at = time.time()
                        self.active_tasks[task.task_id] = task
                        # Update agent load
                        load_increment = 1.0 / agent.max_concurrent_tasks
                        self.registry.update_agent_load(agent.agent_id, load_increment)
                        LOG.info(f"[info] Task {task.task_id} assigned to agent {agent.agent_id}")

                        # Broadcast the task assignment
                        if self.broadcast_callback:
                            await self.broadcast_callback(json.dumps({
                                "type": "task_assigned",
                                "agent_id": agent.agent_id,
                                "task_id": task.task_id,
                                "description": task.description,
                                "timestamp": time.time(),
                            }))

                        return task.task_id
        return None

    def mark_task_completed(self, task_id: str, result: Any, error: Optional[str] = None) -> None:
        """Mark task as completed and update agent metrics"""
        if task_id not in self.active_tasks:
            LOG.warning(f"[warn] Task {task_id} not found in active tasks")
            return

        task = self.active_tasks.pop(task_id)
        task.completed_at = time.time()
        task.result = result
        task.error = error
        task.status = TaskStatus.COMPLETED if error is None else TaskStatus.FAILED

        # Broadcast the task completion
        if self.broadcast_callback:
            asyncio.create_task(self.broadcast_callback(json.dumps({
                "type": "task_completed",
                "task_id": task.task_id,
                "status": task.status.value,
                "result": result,
                "error": error,
                "timestamp": time.time(),
            })))

        # Update agent metrics
        if task.assigned_agent:
            agent = self.registry.get_agent(task.assigned_agent)
            if agent:
                # Update load
                load_decrement = -1.0 / agent.max_concurrent_tasks
                self.registry.update_agent_load(task.assigned_agent, load_decrement)
                # Update performance metrics
                agent.performance_metrics["tasks_completed"] += 1
                completion_time = task.completed_at - task.started_at
                prev_avg = agent.performance_metrics["avg_completion_time"]
                n = agent.performance_metrics["tasks_completed"]
                agent.performance_metrics["avg_completion_time"] = (prev_avg * (n-1) + completion_time) / n
                # Update success rate
                if error is None:
                    prev_rate = agent.performance_metrics["success_rate"]
                    agent.performance_metrics["success_rate"] = (prev_rate * (n-1) + 1.0) / n
                else:
                    prev_rate = agent.performance_metrics["success_rate"]
                    agent.performance_metrics["success_rate"] = (prev_rate * (n-1) + 0.0) / n

        self.completed_tasks[task_id] = task
        LOG.info(f"[info] Task {task_id} completed: {task.status.value}")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "queued": {priority.value: len(tasks) for priority, tasks in self.task_queue.items()},
            "active": len(self.active_tasks),
            "completed": len(self.completed_tasks),
            "total_pending": sum(len(tasks) for tasks in self.task_queue.values())
        }


class ResultAggregator:
    """Merge and cross-validate results from multiple agents"""

    def __init__(self, distributor: TaskDistributor):
        self.distributor = distributor
        self.experiments: Dict[str, ExperimentResult] = {}

    def aggregate_experiment(self, experiment_id: str, task_ids: List[str],
                           merge_fn: Callable[[List[Any]], Dict[str, Any]],
                           validation_fn: Optional[Callable[[Dict[str, Any]], bool]] = None) -> ExperimentResult:
        """Aggregate results from multiple tasks into experiment result"""
        # Collect task results
        task_results = []
        agent_contributions = {}

        for task_id in task_ids:
            task = self.distributor.completed_tasks.get(task_id)
            if not task:
                LOG.warning(f"[warn] Task {task_id} not completed for experiment {experiment_id}")
                continue

            if task.status != TaskStatus.COMPLETED or task.error:
                LOG.warning(f"[warn] Task {task_id} failed, skipping in aggregation")
                continue

            task_results.append(task.result)
            if task.assigned_agent:
                agent_contributions[task.assigned_agent] = task.result

        # Merge results
        merged_data = merge_fn(task_results)

        # Cross-validation
        cross_validation_passed = True
        validation_status = "passed"
        if validation_fn:
            try:
                cross_validation_passed = validation_fn(merged_data)
                validation_status = "passed" if cross_validation_passed else "failed"
            except Exception as e:
                LOG.error(f"[error] Validation failed: {e}")
                validation_status = "error"
                cross_validation_passed = False

        # Compute confidence (based on task success rates and cross-validation)
        confidence = 1.0
        if agent_contributions:
            agent_confidences = []
            for agent_id in agent_contributions.keys():
                agent = self.distributor.registry.get_agent(agent_id)
                if agent:
                    agent_confidences.append(agent.performance_metrics["success_rate"])
            if agent_confidences:
                confidence = sum(agent_confidences) / len(agent_confidences)

        if not cross_validation_passed:
            confidence *= 0.5  # Reduce confidence if validation failed

        result = ExperimentResult(
            experiment_id=experiment_id,
            task_ids=task_ids,
            agent_contributions=agent_contributions,
            merged_data=merged_data,
            validation_status=validation_status,
            confidence=confidence,
            cross_validation_passed=cross_validation_passed
        )

        self.experiments[experiment_id] = result
        LOG.info(f"[info] Experiment {experiment_id} aggregated: {validation_status}, confidence={confidence:.2f}")
        return result

    def statistical_cross_validation(self, results: List[Dict[str, Any]],
                                    key: str, tolerance: float = 0.05) -> bool:
        """Cross-validate results statistically (must agree within tolerance)"""
        values = [r.get(key) for r in results if key in r]
        if len(values) < 2:
            return True  # Can't cross-validate single result

        # Check if all values are numeric
        if not all(isinstance(v, (int, float)) for v in values):
            return True  # Non-numeric data, skip validation

        # Compute mean and check all within tolerance
        mean_val = sum(values) / len(values)
        if mean_val == 0:
            return all(abs(v) < tolerance for v in values)
        else:
            return all(abs(v - mean_val) / abs(mean_val) < tolerance for v in values)


class KnowledgeSharing:
    """Broadcast findings and maintain shared knowledge base"""

    def __init__(self, agent_registry: 'AgentRegistry'):
        self.agent_registry = agent_registry
        self.knowledge_base: Dict[str, Any] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)  # topic -> agent_ids
        self.broadcasts: List[Dict[str, Any]] = []

    def subscribe(self, agent_id: str, topic: str) -> None:
        """Subscribe agent to knowledge topic"""
        self.subscriptions[topic].add(agent_id)
        LOG.info(f"[info] Agent {agent_id} subscribed to topic '{topic}'")

    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """Unsubscribe agent from topic"""
        self.subscriptions[topic].discard(agent_id)

    def publish(self, topic: str, data: Dict[str, Any], source_agent: str) -> None:
        """Publish knowledge to topic"""
        broadcast = {
            "topic": topic,
            "data": data,
            "source_agent": source_agent,
            "timestamp": time.time()
        }
        self.broadcasts.append(broadcast)

        # Add to knowledge base
        if topic not in self.knowledge_base:
            self.knowledge_base[topic] = []
        self.knowledge_base[topic].append(broadcast)

        # Notify subscribers
        subscribers = self.subscriptions.get(topic, set())
        for agent_id in subscribers:
            agent_instance = self.agent_registry.get_agent_instance(agent_id)
            if agent_instance and hasattr(agent_instance, 'process_broadcast'):
                agent_instance.process_broadcast(topic, data)
        
        LOG.info(f"[info] Knowledge published to topic '{topic}' by {source_agent}, {len(subscribers)} subscribers notified")

    def query(self, topic: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query knowledge base"""
        results = self.knowledge_base.get(topic, [])

        # Apply filters
        if filters:
            filtered = []
            for item in results:
                match = True
                for key, value in filters.items():
                    if key not in item["data"] or item["data"][key] != value:
                        match = False
                        break
                if match:
                    filtered.append(item)
            return filtered

        return results

    def get_latest(self, topic: str, n: int = 1) -> List[Dict[str, Any]]:
        """Get latest n broadcasts for topic"""
        results = self.knowledge_base.get(topic, [])
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)[:n]

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of knowledge base"""
        return {
            "topics": list(self.knowledge_base.keys()),
            "total_broadcasts": len(self.broadcasts),
            "subscriptions": {topic: len(agents) for topic, agents in self.subscriptions.items()}
        }


class HiveMind:
    """Main coordinator for multi-agent laboratory system"""

    def __init__(self):
        self.registry = AgentRegistry()
        self.broadcast_callback: Optional[Callable] = None
        self.proposal_callback: Optional[Callable] = None
        self.distributor = TaskDistributor(self.registry, self.broadcast_callback)
        self.aggregator = ResultAggregator(self.distributor)
        self.knowledge = KnowledgeSharing(self.registry)
        self.running = False

    def set_broadcast_callback(self, callback: Callable):
        """Set the callback for broadcasting real-time events."""
        self.broadcast_callback = callback
        self.distributor.broadcast_callback = callback

    def set_proposal_callback(self, callback: Callable):
        """Set the callback for agents to push proposals to the UI."""
        self.proposal_callback = callback

    async def start(self) -> None:
        """Start hive mind coordination"""
        self.running = True
        LOG.info("[info] Hive Mind started")
        # Start task distribution loop
        asyncio.create_task(self._distribution_loop())

    async def stop(self) -> None:
        """Stop hive mind coordination"""
        self.running = False
        LOG.info("[info] Hive Mind stopped")

    async def _distribution_loop(self) -> None:
        """Continuously distribute tasks to agents"""
        while self.running:
            await self.distributor.distribute_next()
            # Check for new knowledge and notify agents
            # This is a simplified polling mechanism. A more robust system
            # might use a dedicated event bus.
            # For now, the publish method directly calls the agents.
            await asyncio.sleep(0.1)  # Check queue every 100ms

    def register_agent(self, agent: Agent, agent_instance: 'Level6AgentInterface' = None) -> None:
        """Register agent in hive mind"""
        self.registry.register_agent(agent, agent_instance)

    def submit_task(self, task: Task) -> str:
        """Submit task for execution"""
        return self.distributor.submit_task(task)

    def submit_experiment(self, experiment_id: str, tasks: List[Task]) -> List[str]:
        """Submit multi-task experiment"""
        LOG.info(f"[info] Submitting experiment {experiment_id} with {len(tasks)} tasks")
        return self.distributor.submit_tasks(tasks)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of hive mind"""
        return {
            "registry": self.registry.get_status_summary(),
            "queue": self.distributor.get_queue_status(),
            "knowledge": self.knowledge.get_summary(),
            "running": self.running
        }

    def export_state(self, filepath: str) -> None:
        """Export hive mind state to JSON"""
        state = {
            "agents": {aid: {
                "agent_id": a.agent_id,
                "agent_type": a.agent_type.value,
                "capabilities": a.capabilities,
                "current_load": a.current_load,
                "performance_metrics": a.performance_metrics
            } for aid, a in self.registry.agents.items()},
            "queue_status": self.distributor.get_queue_status(),
            "knowledge_summary": self.knowledge.get_summary(),
            "timestamp": time.time()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        LOG.info(f"[info] Hive mind state exported to {filepath}")


# Convenience function for creating common agents
def create_standard_agents(
    hive_mind: 'HiveMind | None' = None,
    *,
    auto_register: bool | None = None,
) -> List[Agent]:
    """Create standard set of laboratory agents.

    Parameters
    ----------
    hive_mind:
        Optional HiveMind instance. When provided and ``auto_register`` is enabled,
        agents are registered automatically. When omitted, the caller receives
        unregistered Agent objects to wire up manually (legacy behaviour).
    auto_register:
        Whether to register the created agents with the provided hive mind.
        Defaults to ``True`` when a hive mind is supplied, otherwise ``False``.
    """
    from .agent_interface import create_level6_agent, AgentType

    should_register = auto_register if auto_register is not None else hive_mind is not None

    agent_definitions = [
        (AgentType.PHYSICS, "physics-001", ["mechanics", "thermodynamics", "fluid_dynamics", "electromagnetism"]),
        (AgentType.QUANTUM, "quantum-001", ["quantum_circuits", "quantum_chemistry", "vqe"]),
        (AgentType.MATERIALS, "materials-001", ["tensile_test", "thermal_test", "corrosion_test"]),
        (AgentType.CHEMISTRY, "chemistry-001", ["molecular_dynamics", "reaction_simulation"]),
        (AgentType.ENVIRONMENT, "environment-001", ["temperature_control", "pressure_control"]),
        (AgentType.VALIDATION, "validation-001", ["data_validation", "statistical_testing"]),
    ]
    
    agents = []
    for agent_type, agent_id, capabilities in agent_definitions:
        agent_interface = None
        try:
            agent_interface = create_level6_agent(agent_type, agent_id, hive_mind)
            agent = agent_interface.agent
        except ValueError:
            # Fall back to a basic Agent when no specialised interface exists.
            agent = Agent(agent_id=agent_id, agent_type=agent_type, capabilities=capabilities)

        if hive_mind and should_register:
            hive_mind.register_agent(agent, agent_interface)

        agents.append(agent)
        
    return agents


if __name__ == "__main__":
    # Demo
    async def demo():
        hive = HiveMind()
        await hive.start()

        # Register agents
        for agent in create_standard_agents(hive):
            # The registration is now handled inside create_standard_agents
            pass

        # Submit test tasks
        task1 = Task(
            task_id=str(uuid.uuid4()),
            task_type="tensile_test",
            description="Test carbon fiber tensile strength",
            priority=TaskPriority.HIGH,
            required_capabilities=["tensile_test"],
            parameters={"material": "carbon_fiber", "temperature": 25}
        )

        task2 = Task(
            task_id=str(uuid.uuid4()),
            task_type="thermal_test",
            description="Thermal conductivity measurement",
            priority=TaskPriority.NORMAL,
            required_capabilities=["thermal_test"],
            parameters={"material": "carbon_fiber", "temperature_range": [-50, 200]}
        )

        hive.submit_task(task1)
        hive.submit_task(task2)

        # Wait for distribution
        await asyncio.sleep(1.0)

        # Print status
        status = hive.get_status()
        print(json.dumps(status, indent=2))

        await hive.stop()

    asyncio.run(demo())
