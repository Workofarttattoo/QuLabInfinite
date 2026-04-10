"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Agent Interface - Level-6 Agent API for All Laboratory Departments
Unified interface for autonomous agents with self-improvement and meta-learning
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import time

from .hive_mind_core import Agent, AgentType, Task, TaskStatus, HiveMind
from .semantic_lattice import KnowledgeGraph, ConceptNode, RelationshipEdge, RelationType
from .crystalline_intent import IntentParser, ExperimentDesigner, ParsedIntent
from .temporal_bridge import TemporalBridge, TimeScale

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Internal state of Level-6 agent"""
    agent_id: str
    knowledge_graph: KnowledgeGraph
    learning_rate: float = 0.1
    exploration_rate: float = 0.2
    confidence_threshold: float = 0.8
    performance_history: List[float] = None
    meta_learning_enabled: bool = True

    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []


class Level6AgentInterface(ABC):
    """Abstract interface for Level-6 autonomous agents"""

    def __init__(self, agent: Agent, hive_mind: HiveMind = None):
        self.agent = agent
        self.hive_mind = hive_mind
        self.state = AgentState(
            agent_id=agent.agent_id,
            knowledge_graph=KnowledgeGraph()
        )
        self.intent_parser = IntentParser()
        self.experiment_designer = ExperimentDesigner()
        self.temporal_bridge = TemporalBridge()

    @abstractmethod
    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute assigned task"""
        pass

    @abstractmethod
    def learn_from_result(self, task: Task, result: Dict[str, Any]) -> None:
        """Learn from task result and update knowledge"""
        pass

    def subscribe_to_topic(self, topic: str):
        """Subscribe the agent to a knowledge topic."""
        if self.hive_mind:
            self.hive_mind.knowledge.subscribe(self.agent.agent_id, topic)
            LOG.info(f"Agent {self.agent.agent_id} subscribed to '{topic}'")

    def process_broadcast(self, topic: str, data: Dict[str, Any]):
        """Process a message from a subscribed topic."""
        # Default implementation does nothing; subclasses should override this.
        pass

    def parse_intent(self, query: str) -> ParsedIntent:
        """Parse natural language intent"""
        return self.intent_parser.parse(query)

    def design_experiment(self, intent: ParsedIntent) -> Any:
        """Design experiment from intent"""
        from .crystalline_intent import ExperimentType
        return self.experiment_designer.create_design(intent, ExperimentType.LATIN_HYPERCUBE)

    def record_knowledge(self, concept: ConceptNode) -> None:
        """Record new knowledge in graph"""
        self.state.knowledge_graph.add_node(concept)
        LOG.info(f"[info] Agent {self.agent.agent_id} recorded knowledge: {concept.name}")

    def query_knowledge(self, concept_type: str, filters: Optional[Dict[str, Any]] = None) -> List[ConceptNode]:
        """Query knowledge graph"""
        return self.state.knowledge_graph.query_by_properties(concept_type, filters or {})

    def self_evaluate(self) -> Dict[str, Any]:
        """Evaluate own performance and adjust parameters"""
        if len(self.state.performance_history) < 5:
            return {"status": "insufficient_data"}

        recent_performance = self.state.performance_history[-10:]
        avg_performance = sum(recent_performance) / len(recent_performance)

        # Adjust learning rate based on performance trend
        if len(self.state.performance_history) >= 2:
            trend = recent_performance[-1] - recent_performance[0]
            if trend < 0:  # Performance declining
                self.state.learning_rate *= 1.1  # Increase learning rate
                LOG.info(f"[info] Agent {self.agent.agent_id} increased learning rate to {self.state.learning_rate:.3f}")
            elif trend > 0.1:  # Performance improving rapidly
                self.state.learning_rate *= 0.9  # Decrease learning rate (fine-tune)

        # Adjust exploration rate (epsilon-greedy)
        if avg_performance > 0.8:
            self.state.exploration_rate = max(0.05, self.state.exploration_rate * 0.95)  # Decay exploration
        else:
            self.state.exploration_rate = min(0.5, self.state.exploration_rate * 1.05)  # Increase exploration

        return {
            "avg_performance": avg_performance,
            "learning_rate": self.state.learning_rate,
            "exploration_rate": self.state.exploration_rate,
            "knowledge_nodes": len(self.state.knowledge_graph.nodes),
            "trend": "improving" if trend > 0 else "declining"
        }

    def meta_learn(self, strategy: str, result: float) -> None:
        """Meta-learning: learn about learning strategies"""
        if not self.state.meta_learning_enabled:
            return

        # Record performance of strategy
        self.agent.metadata.setdefault("strategy_performance", {})
        if strategy not in self.agent.metadata["strategy_performance"]:
            self.agent.metadata["strategy_performance"][strategy] = []

        self.agent.metadata["strategy_performance"][strategy].append(result)

        # Identify best strategy
        avg_performance = {
            s: sum(p) / len(p)
            for s, p in self.agent.metadata["strategy_performance"].items()
            if len(p) > 0
        }

        if avg_performance:
            best_strategy = max(avg_performance, key=avg_performance.get)
            self.agent.metadata["best_strategy"] = best_strategy
            LOG.info(f"[info] Agent {self.agent.agent_id} best strategy: {best_strategy} ({avg_performance[best_strategy]:.3f})")

    def propose_experiment(self, goal: str) -> Dict[str, Any]:
        """Autonomously propose experiment to achieve goal"""
        # Parse goal
        intent = self.parse_intent(goal)

        # Design experiment
        design = self.design_experiment(intent)

        # Estimate resources
        from .crystalline_intent import ResourceEstimator
        estimator = ResourceEstimator()
        resources = estimator.estimate(design)

        proposal = {
            "intent": {
                "type": intent.experiment_type,
                "materials": intent.materials,
                "properties": intent.properties,
                "confidence": intent.confidence
            },
            "design": {
                "design_id": design.design_id,
                "type": design.design_type.value,
                "num_runs": design.num_runs,
                "estimated_duration": design.estimated_duration
            },
            "resources": resources
        }

        # Push the proposal to the UI if a callback is available
        if self.hive_mind and self.hive_mind.proposal_callback:
            import asyncio
            asyncio.create_task(self.hive_mind.proposal_callback(self.agent.agent_id, proposal))

        return proposal

    def export_knowledge(self, filepath: str) -> None:
        """Export agent's knowledge graph"""
        self.state.knowledge_graph.save(filepath)

    def import_knowledge(self, filepath: str) -> None:
        """Import knowledge from file"""
        self.state.knowledge_graph.load(filepath)

    def get_capabilities_report(self) -> Dict[str, Any]:
        """Generate report of agent capabilities"""
        return {
            "agent_id": self.agent.agent_id,
            "agent_type": self.agent.agent_type.value,
            "capabilities": self.agent.capabilities,
            "current_load": self.agent.current_load,
            "performance_metrics": self.agent.performance_metrics,
            "knowledge_graph_size": len(self.state.knowledge_graph.nodes),
            "learning_parameters": {
                "learning_rate": self.state.learning_rate,
                "exploration_rate": self.state.exploration_rate,
                "confidence_threshold": self.state.confidence_threshold
            }
        }


class PhysicsAgent(Level6AgentInterface):
    """Physics simulation agent"""

    def __init__(self, agent: Agent, hive_mind: HiveMind = None):
        super().__init__(agent, hive_mind)
        # Automatically subscribe to the hearing channel on creation
        self.subscribe_to_topic("hearing_channel")

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute physics simulation task"""
        LOG.info(f"[info] PhysicsAgent executing: {task.description}")

        task_type = task.task_type
        params = task.parameters

        if task_type == "mechanics":
            result = self._simulate_mechanics(params)
        elif task_type == "thermodynamics":
            result = self._simulate_thermodynamics(params)
        elif task_type == "fluid_dynamics":
            result = self._simulate_fluid_dynamics(params)
        elif task_type == "electromagnetism":
            result = self._simulate_electromagnetism(params)
        else:
            result = {"error": f"Unknown task type: {task_type}"}

        # Learn from result
        self.learn_from_result(task, result)

        return result

    def _simulate_mechanics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate mechanical system"""
        # Simplified simulation
        force = params.get("force", 100.0)  # N
        mass = params.get("mass", 10.0)  # kg
        acceleration = force / mass

        return {
            "acceleration": acceleration,
            "force": force,
            "mass": mass,
            "simulation_time": 0.1,
            "confidence": 0.95
        }

    def _simulate_thermodynamics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate thermodynamic process"""
        temperature = params.get("temperature", 300.0)  # K
        pressure = params.get("pressure", 101325.0)  # Pa
        volume = params.get("volume", 1.0)  # m³

        # Ideal gas law
        n_moles = (pressure * volume) / (8.314 * temperature)

        return {
            "temperature": temperature,
            "pressure": pressure,
            "volume": volume,
            "moles": n_moles,
            "confidence": 0.9
        }

    def _simulate_fluid_dynamics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate fluid flow"""
        wind_speed = params.get("wind_speed", 10.0)  # m/s
        density = params.get("density", 1.225)  # kg/m³
        area = params.get("area", 1.0)  # m²

        # Dynamic pressure
        dynamic_pressure = 0.5 * density * wind_speed**2

        # Drag force (assuming Cd = 1.0)
        drag_force = dynamic_pressure * area

        return {
            "wind_speed": wind_speed,
            "dynamic_pressure": dynamic_pressure,
            "drag_force": drag_force,
            "confidence": 0.85
        }

    def _simulate_electromagnetism(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate electromagnetic field"""
        current = params.get("current", 1.0)  # A
        resistance = params.get("resistance", 10.0)  # Ohms

        voltage = current * resistance
        power = voltage * current

        return {
            "voltage": voltage,
            "current": current,
            "resistance": resistance,
            "power": power,
            "confidence": 0.95
        }

    def learn_from_result(self, task: Task, result: Dict[str, Any]) -> None:
        """Learn from simulation result"""
        # Record performance
        confidence = result.get("confidence", 0.5)
        self.state.performance_history.append(confidence)

        # Add to knowledge graph
        concept = ConceptNode(
            node_id=f"result_{task.task_id}",
            concept_type="result",
            name=f"{task.task_type}_result",
            properties=result,
            confidence=confidence
        )
        self.record_knowledge(concept)

        # Self-evaluate
        self.self_evaluate()

    def process_broadcast(self, topic: str, data: Dict[str, Any]):
        """React to messages on the hearing channel."""
        if topic == "hearing_channel":
            text = data.get("text", "").lower()
            LOG.info(f"PhysicsAgent heard: '{text}'")
            
            # Simple keyword-based reaction
            if "simulate wind" in text or "wind load" in text:
                self.propose_experiment("Simulate wind load on a standard plate")


class QuantumAgent(Level6AgentInterface):
    """Quantum simulation agent"""

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute quantum simulation task"""
        LOG.info(f"[info] QuantumAgent executing: {task.description}")

        task_type = task.task_type
        params = task.parameters

        if task_type == "quantum_circuit":
            result = self._simulate_quantum_circuit(params)
        elif task_type == "quantum_chemistry":
            result = self._simulate_quantum_chemistry(params)
        else:
            result = {"error": f"Unknown task type: {task_type}"}

        self.learn_from_result(task, result)
        return result

    def _simulate_quantum_circuit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate quantum circuit"""
        num_qubits = params.get("num_qubits", 5)

        # Simplified simulation
        fidelity = 0.95 ** num_qubits  # Decoherence effect

        return {
            "num_qubits": num_qubits,
            "fidelity": fidelity,
            "execution_time": num_qubits * 0.1,
            "confidence": 0.9
        }

    def _simulate_quantum_chemistry(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate molecular energy with VQE"""
        molecule = params.get("molecule", "H2")
        basis_set = params.get("basis_set", "sto-3g")

        # Simplified energy calculation
        energy = -1.13 if molecule == "H2" else -75.0  # Hartree

        return {
            "molecule": molecule,
            "basis_set": basis_set,
            "energy": energy,
            "confidence": 0.85
        }

    def learn_from_result(self, task: Task, result: Dict[str, Any]) -> None:
        """Learn from quantum result"""
        confidence = result.get("confidence", 0.5)
        self.state.performance_history.append(confidence)

        concept = ConceptNode(
            node_id=f"quantum_result_{task.task_id}",
            concept_type="result",
            name=f"{task.task_type}_result",
            properties=result,
            confidence=confidence
        )
        self.record_knowledge(concept)


class MaterialsAgent(Level6AgentInterface):
    """Materials testing agent"""

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute materials testing task"""
        LOG.info(f"[info] MaterialsAgent executing: {task.description}")

        task_type = task.task_type
        params = task.parameters

        if task_type == "tensile_test":
            result = self._tensile_test(params)
        elif task_type == "thermal_test":
            result = self._thermal_test(params)
        elif task_type == "corrosion_test":
            result = self._corrosion_test(params)
        else:
            result = {"error": f"Unknown task type: {task_type}"}

        self.learn_from_result(task, result)
        return result

    def _tensile_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate tensile test"""
        material = params.get("material", "steel")
        temperature = params.get("temperature", 25)

        # Simplified material properties
        tensile_strength = 500 if material == "steel" else 300  # MPa
        yield_strength = tensile_strength * 0.7

        # Temperature correction
        temp_factor = 1.0 - (temperature - 25) * 0.001
        tensile_strength *= temp_factor
        yield_strength *= temp_factor

        return {
            "material": material,
            "tensile_strength": tensile_strength,
            "yield_strength": yield_strength,
            "elongation": 0.25,
            "temperature": temperature,
            "confidence": 0.9
        }

    def _thermal_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Thermal conductivity test"""
        material = params.get("material", "copper")

        conductivity = 400 if material == "copper" else 50  # W/m·K

        return {
            "material": material,
            "thermal_conductivity": conductivity,
            "specific_heat": 385,
            "confidence": 0.88
        }

    def _corrosion_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Corrosion resistance test"""
        material = params.get("material", "steel")
        environment = params.get("environment", "saltwater")

        corrosion_rate = 0.5 if material == "stainless_steel" else 5.0  # mm/year

        return {
            "material": material,
            "environment": environment,
            "corrosion_rate": corrosion_rate,
            "confidence": 0.8
        }

    def learn_from_result(self, task: Task, result: Dict[str, Any]) -> None:
        """Learn from materials test"""
        confidence = result.get("confidence", 0.5)
        self.state.performance_history.append(confidence)

        # Add material properties to knowledge graph
        material = result.get("material")
        if material:
            concept = ConceptNode(
                node_id=f"material_{material}_{task.task_id}",
                concept_type="material",
                name=material,
                properties=result,
                confidence=confidence
            )
            self.record_knowledge(concept)


def create_level6_agent(agent_type: AgentType, agent_id: str, hive_mind: HiveMind = None) -> Level6AgentInterface:
    """Factory function to create Level-6 agents"""
    # Create base agent
    if agent_type == AgentType.PHYSICS:
        capabilities = ["mechanics", "thermodynamics", "fluid_dynamics", "electromagnetism"]
        base_agent = Agent(agent_id=agent_id, agent_type=agent_type, capabilities=capabilities)
        return PhysicsAgent(base_agent, hive_mind)

    elif agent_type == AgentType.QUANTUM:
        capabilities = ["quantum_circuit", "quantum_chemistry", "vqe"]
        base_agent = Agent(agent_id=agent_id, agent_type=agent_type, capabilities=capabilities)
        return QuantumAgent(base_agent, hive_mind)

    elif agent_type == AgentType.MATERIALS:
        capabilities = ["tensile_test", "thermal_test", "corrosion_test", "fatigue_test"]
        base_agent = Agent(agent_id=agent_id, agent_type=agent_type, capabilities=capabilities)
        return MaterialsAgent(base_agent, hive_mind)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


if __name__ == "__main__":
    import asyncio

    async def demo():
        # Create agents
        physics_agent = create_level6_agent(AgentType.PHYSICS, "physics-demo-001")
        materials_agent = create_level6_agent(AgentType.MATERIALS, "materials-demo-001")

        # Test physics agent
        task1 = Task(
            task_id="task_001",
            task_type="fluid_dynamics",
            description="Simulate wind load",
            priority=TaskPriority.NORMAL,
            required_capabilities=["fluid_dynamics"],
            parameters={"wind_speed": 30, "area": 2.0}
        )

        result1 = await physics_agent.execute_task(task1)
        print(f"Physics result: {json.dumps(result1, indent=2)}")

        # Test materials agent
        task2 = Task(
            task_id="task_002",
            task_type="tensile_test",
            description="Test carbon fiber",
            priority=TaskPriority.NORMAL,
            required_capabilities=["tensile_test"],
            parameters={"material": "carbon_fiber", "temperature": 200}
        )

        result2 = await materials_agent.execute_task(task2)
        print(f"Materials result: {json.dumps(result2, indent=2)}")

        # Self-evaluation
        eval1 = physics_agent.self_evaluate()
        print(f"Physics agent evaluation: {json.dumps(eval1, indent=2)}")

        # Propose experiment
        proposal = materials_agent.propose_experiment("Find lightweight corrosion-resistant alloy")
        print(f"Experiment proposal: {json.dumps(proposal, indent=2)}")

    asyncio.run(demo())
