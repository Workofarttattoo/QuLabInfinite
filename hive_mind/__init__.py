"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Hive Mind Coordination - Multi-Agent Laboratory System for QuLabInfinite
"""

from .hive_mind_core import (
    HiveMind,
    Agent,
    AgentType,
    Task,
    TaskPriority,
    TaskStatus,
    AgentRegistry,
    TaskDistributor,
    ResultAggregator,
    KnowledgeSharing,
    ExperimentResult,
    create_standard_agents
)

from .semantic_lattice import (
    KnowledgeGraph,
    ConceptNode,
    RelationshipEdge,
    RelationType,
    PropertyTensor,
    InferenceEngine
)

from .crystalline_intent import (
    IntentParser,
    ExperimentDesigner,
    ResourceEstimator,
    SuccessCriteria,
    RiskAssessment,
    ParsedIntent,
    ExperimentDesign,
    ExperimentType,
    OptimizationGoal,
    Parameter,
    Objective,
    Constraint
)

from .temporal_bridge import (
    TemporalBridge,
    TimeScaleManager,
    TemporalSynchronization,
    EventDetection,
    CheckpointManager,
    AcceleratedDynamics,
    TimeScale,
    TimePoint,
    Event,
    Checkpoint
)

from .orchestrator import (
    Orchestrator,
    WorkflowEngine,
    ResultsValidator,
    MultiPhysicsExperiment,
    WorkflowNode,
    WorkflowEdge,
    WorkflowStatus
)

from .agent_interface import (
    Level6AgentInterface,
    PhysicsAgent,
    QuantumAgent,
    MaterialsAgent,
    AgentState,
    create_level6_agent
)

__version__ = "1.0.0"
__all__ = [
    # Core
    "HiveMind",
    "Agent",
    "AgentType",
    "Task",
    "TaskPriority",
    "TaskStatus",
    "AgentRegistry",
    "TaskDistributor",
    "ResultAggregator",
    "KnowledgeSharing",
    "ExperimentResult",
    "create_standard_agents",

    # Semantic Lattice
    "KnowledgeGraph",
    "ConceptNode",
    "RelationshipEdge",
    "RelationType",
    "PropertyTensor",
    "InferenceEngine",

    # Crystalline Intent
    "IntentParser",
    "ExperimentDesigner",
    "ResourceEstimator",
    "SuccessCriteria",
    "RiskAssessment",
    "ParsedIntent",
    "ExperimentDesign",
    "ExperimentType",
    "OptimizationGoal",
    "Parameter",
    "Objective",
    "Constraint",

    # Temporal Bridge
    "TemporalBridge",
    "TimeScaleManager",
    "TemporalSynchronization",
    "EventDetection",
    "CheckpointManager",
    "AcceleratedDynamics",
    "TimeScale",
    "TimePoint",
    "Event",
    "Checkpoint",

    # Orchestrator
    "Orchestrator",
    "WorkflowEngine",
    "ResultsValidator",
    "MultiPhysicsExperiment",
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowStatus",

    # Agent Interface
    "Level6AgentInterface",
    "PhysicsAgent",
    "QuantumAgent",
    "MaterialsAgent",
    "AgentState",
    "create_level6_agent"
]
