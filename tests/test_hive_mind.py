"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Test Suite for Hive Mind Coordination System
"""

import unittest
import asyncio
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hive_mind import (
    HiveMind, Agent, AgentType, Task, TaskPriority, TaskStatus,
    KnowledgeGraph, ConceptNode, RelationshipEdge, RelationType,
    IntentParser, ExperimentDesigner, ExperimentType,
    TemporalBridge, TimeScale,
    Orchestrator, MultiPhysicsExperiment, WorkflowNode,
    create_level6_agent, PhysicsAgent, MaterialsAgent,
    create_standard_agents
)


class TestHiveMindCore(unittest.TestCase):
    """Test HiveMind core functionality"""

    def setUp(self):
        self.hive = HiveMind()

    def test_agent_registration(self):
        """Test agent registration"""
        agent = Agent(
            agent_id="test-001",
            agent_type=AgentType.PHYSICS,
            capabilities=["mechanics", "thermodynamics"]
        )
        self.hive.register_agent(agent)

        status = self.hive.get_status()
        self.assertEqual(status["registry"]["total_agents"], 1)
        self.assertEqual(status["registry"]["by_type"]["physics"], 1)

    def test_task_submission(self):
        """Test task submission and queueing"""
        agent = Agent(
            agent_id="test-001",
            agent_type=AgentType.PHYSICS,
            capabilities=["mechanics"]
        )
        self.hive.register_agent(agent)

        task = Task(
            task_id="task-001",
            task_type="mechanics",
            description="Test mechanics simulation",
            priority=TaskPriority.HIGH,
            required_capabilities=["mechanics"],
            parameters={"force": 100}
        )

        task_id = self.hive.submit_task(task)
        self.assertEqual(task_id, "task-001")

        queue_status = self.hive.distributor.get_queue_status()
        self.assertGreater(queue_status["total_pending"], 0)

    def test_knowledge_sharing(self):
        """Test knowledge sharing between agents"""
        self.hive.knowledge.subscribe("agent-001", "discoveries")

        self.hive.knowledge.publish(
            "discoveries",
            {"finding": "new material property"},
            "agent-002"
        )

        latest = self.hive.knowledge.get_latest("discoveries", n=1)
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["source_agent"], "agent-002")


class TestSemanticLattice(unittest.TestCase):
    """Test knowledge graph functionality"""

    def setUp(self):
        self.kg = KnowledgeGraph()

    def test_node_addition(self):
        """Test adding nodes to knowledge graph"""
        node = ConceptNode(
            node_id="mat_001",
            concept_type="material",
            name="Steel",
            properties={"strength": 500, "density": 8.0}
        )
        self.kg.add_node(node)

        retrieved = self.kg.get_node("mat_001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Steel")

    def test_edge_addition(self):
        """Test adding edges to knowledge graph"""
        node1 = ConceptNode(node_id="mat_001", concept_type="material", name="Steel", properties={})
        node2 = ConceptNode(node_id="mat_002", concept_type="material", name="Aluminum", properties={})

        self.kg.add_node(node1)
        self.kg.add_node(node2)

        edge = RelationshipEdge(
            edge_id="edge_001",
            source_node="mat_001",
            target_node="mat_002",
            relation_type=RelationType.SIMILARITY,
            strength=0.8
        )
        self.kg.add_edge(edge)

        edges = self.kg.get_edges_between("mat_001", "mat_002")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relation_type, RelationType.SIMILARITY)

    def test_similarity_search(self):
        """Test finding similar nodes"""
        # Add nodes with embeddings
        for i in range(5):
            node = ConceptNode(
                node_id=f"mat_{i:03d}",
                concept_type="material",
                name=f"Material_{i}",
                properties={"property": i},
                embedding=np.random.randn(128)
            )
            self.kg.add_node(node)

        similar = self.kg.find_similar_nodes("mat_000", top_k=3)
        self.assertEqual(len(similar), 3)
        # Check that returned format is (node_id, similarity)
        self.assertIsInstance(similar[0], tuple)
        self.assertEqual(len(similar[0]), 2)

    def test_property_query(self):
        """Test querying by properties"""
        node1 = ConceptNode(
            node_id="mat_001",
            concept_type="material",
            name="Steel",
            properties={"strength": 500, "type": "metal"}
        )
        node2 = ConceptNode(
            node_id="mat_002",
            concept_type="material",
            name="Aluminum",
            properties={"strength": 300, "type": "metal"}
        )

        self.kg.add_node(node1)
        self.kg.add_node(node2)

        results = self.kg.query_by_properties("material", {"type": "metal"})
        self.assertEqual(len(results), 2)


class TestCrystallineIntent(unittest.TestCase):
    """Test intent parsing and experiment design"""

    def setUp(self):
        self.parser = IntentParser()
        self.designer = ExperimentDesigner()

    def test_intent_parsing(self):
        """Test parsing natural language intent"""
        query = "Optimize lightweight corrosion-resistant alloy with strength > 500 MPa at 200°C"
        intent = self.parser.parse(query)

        self.assertEqual(intent.experiment_type, "optimize")
        self.assertIn("strength", intent.properties)
        self.assertIn("temperature", intent.conditions)
        self.assertEqual(intent.conditions["temperature"], 200)

    def test_material_extraction(self):
        """Test extracting materials from query"""
        query = "Test carbon fiber and aluminum alloy under stress"
        intent = self.parser.parse(query)

        self.assertGreater(len(intent.materials), 0)

    def test_experiment_design(self):
        """Test generating experiment design"""
        intent = self.parser.parse("Optimize battery performance")
        design = self.designer.create_design(intent, ExperimentType.LATIN_HYPERCUBE, num_runs=50)

        self.assertEqual(design.design_type, ExperimentType.LATIN_HYPERCUBE)
        self.assertEqual(design.num_runs, 50)
        self.assertEqual(design.run_matrix.shape[0], 50)

    def test_full_factorial_design(self):
        """Test full factorial design generation"""
        from hive_mind.crystalline_intent import Parameter

        params = [
            Parameter(name="temperature", param_type="continuous", min_value=0, max_value=100),
            Parameter(name="pressure", param_type="continuous", min_value=0, max_value=10)
        ]

        design_matrix = self.designer.design_full_factorial(params, levels=3)
        self.assertEqual(design_matrix.shape[0], 9)  # 3^2 = 9 runs


class TestTemporalBridge(unittest.TestCase):
    """Test time-scale management"""

    def setUp(self):
        self.bridge = TemporalBridge()

    def test_time_advancement(self):
        """Test advancing simulation time"""
        initial_time = self.bridge.time_manager.current_time
        self.bridge.time_manager.advance(100, TimeScale.MILLISECOND)

        new_time = self.bridge.time_manager.current_time
        self.assertGreater(new_time, initial_time)
        self.assertAlmostEqual(new_time, initial_time + 0.1, places=5)  # 100 ms = 0.1 s

    def test_time_scale_conversion(self):
        """Test time scale conversions"""
        self.bridge.time_manager.set_time(1.0, TimeScale.HOUR)

        seconds = self.bridge.time_manager.get_time(TimeScale.SECOND)
        self.assertAlmostEqual(seconds, 3600.0, places=1)

    def test_event_scheduling(self):
        """Test event scheduling and detection"""
        from hive_mind.temporal_bridge import Event

        event = Event(
            event_id="test_event",
            time=5.0,
            event_type="phase_transition"
        )

        self.bridge.event_detector.schedule_event(event)

        # Advance past event time
        self.bridge.time_manager.set_time(10.0, TimeScale.SECOND)
        triggered = self.bridge.event_detector.check_events()

        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0].event_id, "test_event")

    def test_checkpoint_save_restore(self):
        """Test checkpoint save and restore"""
        state = {"temperature": 300, "pressure": 1.0}

        # Save checkpoint
        self.bridge.checkpoint_manager.save_checkpoint("cp1", state)

        # Advance time
        self.bridge.time_manager.advance(100, TimeScale.SECOND)

        # Restore checkpoint
        restored_state = self.bridge.checkpoint_manager.restore_checkpoint("cp1")

        self.assertEqual(restored_state, state)
        self.assertLess(self.bridge.time_manager.current_time, 100)  # Time rolled back


class TestOrchestrator(unittest.TestCase):
    """Test multi-physics orchestration"""

    def setUp(self):
        self.orchestrator = Orchestrator()

    def test_aerogel_experiment_creation(self):
        """Test creating aerogel multi-department experiment"""
        experiment = self.orchestrator.create_aerogel_experiment()

        self.assertIsInstance(experiment, MultiPhysicsExperiment)
        self.assertGreater(len(experiment.workflow), 0)
        self.assertGreater(len(experiment.edges), 0)
        self.assertIn("materials", experiment.departments)
        self.assertIn("physics", experiment.departments)

    def test_workflow_node_structure(self):
        """Test workflow node structure"""
        experiment = self.orchestrator.create_aerogel_experiment()

        # Check that nodes have proper structure
        for node_id, node in experiment.workflow.items():
            self.assertIsInstance(node, WorkflowNode)
            self.assertIsInstance(node.node_id, str)
            self.assertIsInstance(node.description, str)


class TestAgentInterface(unittest.TestCase):
    """Test Level-6 agent interface"""

    def setUp(self):
        self.physics_agent = create_level6_agent(AgentType.PHYSICS, "test-physics-001")
        self.materials_agent = create_level6_agent(AgentType.MATERIALS, "test-materials-001")

    def test_agent_creation(self):
        """Test creating Level-6 agents"""
        self.assertIsInstance(self.physics_agent, PhysicsAgent)
        self.assertEqual(self.physics_agent.agent.agent_id, "test-physics-001")

    def test_intent_parsing(self):
        """Test agent intent parsing"""
        intent = self.physics_agent.parse_intent("Simulate wind load at 50 mph")

        self.assertIsNotNone(intent)
        self.assertGreater(intent.confidence, 0)

    def test_experiment_proposal(self):
        """Test autonomous experiment proposal"""
        proposal = self.materials_agent.propose_experiment("Find corrosion-resistant material")

        self.assertIn("intent", proposal)
        self.assertIn("design", proposal)
        self.assertIn("resources", proposal)

    def test_task_execution(self):
        """Test agent task execution"""
        async def run_test():
            task = Task(
                task_id="test_task",
                task_type="fluid_dynamics",
                description="Simulate airflow",
                priority=TaskPriority.NORMAL,
                required_capabilities=["fluid_dynamics"],
                parameters={"wind_speed": 30}
            )

            result = await self.physics_agent.execute_task(task)

            self.assertIn("drag_force", result)
            self.assertGreater(result["confidence"], 0)

        asyncio.run(run_test())

    def test_knowledge_recording(self):
        """Test agent knowledge recording"""
        concept = ConceptNode(
            node_id="test_concept",
            concept_type="material",
            name="Test Material",
            properties={"strength": 500}
        )

        self.physics_agent.record_knowledge(concept)

        # Check that knowledge was recorded
        nodes = self.physics_agent.query_knowledge("material")
        self.assertGreater(len(nodes), 0)

    def test_self_evaluation(self):
        """Test agent self-evaluation"""
        # Add some performance history
        for i in range(10):
            self.physics_agent.state.performance_history.append(0.8 + i * 0.01)

        evaluation = self.physics_agent.self_evaluate()

        self.assertIn("avg_performance", evaluation)
        self.assertIn("learning_rate", evaluation)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""

    def test_multi_agent_experiment(self):
        """Test complete multi-agent experiment workflow"""
        async def run_test():
            hive = HiveMind()
            await hive.start()

            # Register standard agents
            for agent in create_standard_agents():
                hive.register_agent(agent)

            # Submit tasks
            tasks = [
                Task(
                    task_id=f"task_{i}",
                    task_type="test",
                    description=f"Test task {i}",
                    priority=TaskPriority.NORMAL,
                    required_capabilities=[],
                    parameters={}
                )
                for i in range(5)
            ]

            for task in tasks:
                hive.submit_task(task)

            # Wait briefly for distribution
            await asyncio.sleep(0.5)

            status = hive.get_status()
            self.assertGreater(status["registry"]["total_agents"], 0)

            await hive.stop()

        asyncio.run(run_test())

    def test_knowledge_graph_workflow(self):
        """Test knowledge graph construction and querying"""
        kg = KnowledgeGraph()

        # Add multiple materials
        materials = [
            ("steel", {"strength": 500, "density": 8.0, "cost": 10}),
            ("aluminum", {"strength": 300, "density": 2.7, "cost": 15}),
            ("titanium", {"strength": 900, "density": 4.5, "cost": 100})
        ]

        for i, (name, props) in enumerate(materials):
            node = ConceptNode(
                node_id=f"mat_{i}",
                concept_type="material",
                name=name,
                properties=props
            )
            kg.add_node(node)

        # Query high strength materials
        results = kg.get_nodes_by_type("material")
        high_strength = [n for n in results if n.properties.get("strength", 0) > 400]

        self.assertGreater(len(high_strength), 0)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestHiveMindCore))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticLattice))
    suite.addTests(loader.loadTestsFromTestCase(TestCrystallineIntent))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestOrchestrator))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
