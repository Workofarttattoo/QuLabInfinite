"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Hive Mind Examples - Demonstration of Multi-Agent Coordination
"""

import asyncio
import json
import numpy as np

from hive_mind import (
    HiveMind, create_standard_agents, Task, TaskPriority,
    KnowledgeGraph, ConceptNode, RelationType, RelationshipEdge,
    IntentParser, ExperimentDesigner, ExperimentType, ResourceEstimator,
    TemporalBridge, TimeScale, Event,
    Orchestrator,
    create_level6_agent, AgentType
)


def example_1_basic_hive_mind():
    """Example 1: Basic Hive Mind with agent registration and task distribution"""
    print("=" * 80)
    print("EXAMPLE 1: Basic Hive Mind Operation")
    print("=" * 80)

    async def run():
        # Create hive mind
        hive = HiveMind()
        await hive.start()

        # Register standard agents
        agents = create_standard_agents()
        for agent in agents:
            hive.register_agent(agent)

        print(f"\nRegistered {len(agents)} agents")

        # Submit tasks
        tasks = [
            Task(
                task_id=f"task_{i}",
                task_type="test",
                description=f"Test experiment {i}",
                priority=TaskPriority.NORMAL if i % 2 == 0 else TaskPriority.HIGH,
                required_capabilities=[],
                parameters={"experiment_id": i}
            )
            for i in range(10)
        ]

        for task in tasks:
            hive.submit_task(task)

        print(f"Submitted {len(tasks)} tasks")

        # Wait for distribution
        await asyncio.sleep(1.0)

        # Get status
        status = hive.get_status()
        print(f"\nHive Mind Status:")
        print(json.dumps(status, indent=2))

        await hive.stop()

    asyncio.run(run())


def example_2_knowledge_graph():
    """Example 2: Knowledge Graph with material properties and inference"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Knowledge Graph Construction and Querying")
    print("=" * 80)

    kg = KnowledgeGraph()

    # Add materials
    materials = [
        ("AISI_304_Steel", {"tensile_strength": 515, "yield_strength": 205, "density": 8.0, "corrosion_resistance": "excellent"}),
        ("Aluminum_6061", {"tensile_strength": 310, "yield_strength": 276, "density": 2.7, "corrosion_resistance": "good"}),
        ("Titanium_Ti6Al4V", {"tensile_strength": 950, "yield_strength": 880, "density": 4.43, "corrosion_resistance": "excellent"}),
        ("Carbon_Fiber", {"tensile_strength": 4000, "yield_strength": 3500, "density": 1.6, "corrosion_resistance": "excellent"}),
    ]

    print("\nAdding materials to knowledge graph...")
    for i, (name, props) in enumerate(materials):
        node = ConceptNode(
            node_id=f"mat_{i:03d}",
            concept_type="material",
            name=name,
            properties=props,
            embedding=np.random.randn(128)
        )
        kg.add_node(node)
        print(f"  Added: {name}")

    # Add relationships
    print("\nAdding relationships...")
    relationships = [
        ("mat_000", "mat_001", RelationType.SUBSTITUTION, 0.7, "lightweight_applications"),
        ("mat_002", "mat_003", RelationType.SIMILARITY, 0.8, "high_strength_materials"),
    ]

    for i, (source, target, rel_type, strength, use_case) in enumerate(relationships):
        edge = RelationshipEdge(
            edge_id=f"edge_{i:03d}",
            source_node=source,
            target_node=target,
            relation_type=rel_type,
            strength=strength,
            properties={"use_case": use_case}
        )
        kg.add_edge(edge)
        print(f"  Added: {source} --{rel_type.value}--> {target}")

    # Query similar materials
    print("\nFinding materials similar to Steel...")
    similar = kg.find_similar_nodes("mat_000", top_k=3)
    for node_id, similarity in similar:
        node = kg.get_node(node_id)
        print(f"  {node.name}: similarity={similarity:.3f}")

    # Query by properties
    print("\nQuerying materials with excellent corrosion resistance...")
    results = kg.query_by_properties("material", {"corrosion_resistance": "excellent"})
    for node in results:
        print(f"  {node.name}: strength={node.properties['tensile_strength']} MPa")

    # Inference
    print("\nTesting inference engine...")
    from hive_mind.semantic_lattice import InferenceEngine
    inference = InferenceEngine(kg)

    # Predict property
    predicted_strength, confidence = inference.predict_property("mat_001", "tensile_strength")
    print(f"  Predicted aluminum tensile strength: {predicted_strength:.1f} MPa (confidence: {confidence:.2f})")

    # Analogical reasoning
    print("\nAnalogical reasoning test...")
    # If steel:strong :: aluminum:?, predict ?
    analog = inference.analogical_reasoning(("mat_000", "mat_002"), "mat_001")
    if analog:
        analog_node = kg.get_node(analog)
        print(f"  Steel:Titanium :: Aluminum:? → {analog_node.name if analog_node else analog}")


def example_3_crystalline_intent():
    """Example 3: Intent parsing and experiment design"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Crystalline Intent - NLP Experiment Design")
    print("=" * 80)

    parser = IntentParser()
    designer = ExperimentDesigner()
    estimator = ResourceEstimator()

    # Test queries
    queries = [
        "Find lightweight corrosion-resistant alloy with tensile strength > 500 MPa",
        "Optimize battery electrolyte for maximum ionic conductivity at 60°C",
        "Test carbon fiber thermal stability under 1000°C for 24 hours",
        "Compare aluminum alloys for aerospace applications"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 80)

        # Parse intent
        intent = parser.parse(query)
        print(f"  Experiment Type: {intent.experiment_type}")
        print(f"  Materials: {intent.materials}")
        print(f"  Properties: {intent.properties}")
        print(f"  Conditions: {intent.conditions}")
        print(f"  Objectives: {[o.name for o in intent.objectives]}")
        print(f"  Parse Confidence: {intent.confidence:.2f}")

        # Design experiment
        design = designer.create_design(intent, ExperimentType.LATIN_HYPERCUBE, num_runs=50)
        print(f"\n  Experiment Design:")
        print(f"    Design Type: {design.design_type.value}")
        print(f"    Number of Runs: {design.num_runs}")
        print(f"    Parameters: {[p.name for p in design.parameters]}")

        # Estimate resources
        resources = estimator.estimate(design)
        print(f"\n  Resource Estimate:")
        print(f"    CPU Cores: {resources['cpu_cores']}")
        print(f"    RAM: {resources['ram_mb']} MB")
        print(f"    ETA: {resources['eta_hours']:.2f} hours")
        print(f"    Cost-Benefit Ratio: {resources['cost_benefit_ratio']:.2f}")


def example_4_temporal_bridge():
    """Example 4: Multi-scale temporal simulation"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Temporal Bridge - Multi-Scale Simulation")
    print("=" * 80)

    bridge = TemporalBridge()

    # Schedule events
    print("\nScheduling events...")
    events = [
        Event(event_id="bond_vibration", time=1e-15, event_type="molecular", data={"type": "vibration"}),
        Event(event_id="phase_transition", time=1e-9, event_type="phase_change", data={"from": "solid", "to": "liquid"}),
        Event(event_id="crack_initiation", time=1e-3, event_type="mechanical", data={"stress": 500}),
        Event(event_id="corrosion_visible", time=86400, event_type="chemical", data={"rate": 0.5})
    ]

    for event in events:
        bridge.event_detector.schedule_event(event)
        print(f"  Scheduled: {event.event_id} at t={event.time}s")

    # Simulate across time scales
    print("\nRunning multi-scale simulation...")

    # Femtosecond scale
    print("\n  Femtosecond scale (molecular vibrations)...")
    bridge.time_manager.set_time(100, TimeScale.FEMTOSECOND)
    triggered = bridge.event_detector.check_events()
    print(f"    Current time: {bridge.time_manager.get_time(TimeScale.FEMTOSECOND):.1f} fs")
    print(f"    Triggered events: {[e.event_id for e in triggered]}")

    # Nanosecond scale
    print("\n  Nanosecond scale (phase transitions)...")
    bridge.time_manager.set_time(10, TimeScale.NANOSECOND)
    triggered = bridge.event_detector.check_events()
    print(f"    Current time: {bridge.time_manager.get_time(TimeScale.NANOSECOND):.1f} ns")
    print(f"    Triggered events: {[e.event_id for e in triggered]}")

    # Day scale
    print("\n  Day scale (long-term corrosion)...")
    bridge.time_manager.set_time(2, TimeScale.DAY)
    triggered = bridge.event_detector.check_events()
    print(f"    Current time: {bridge.time_manager.get_time(TimeScale.DAY):.1f} days")
    print(f"    Triggered events: {[e.event_id for e in triggered]}")

    # Accelerated dynamics
    print("\n  Testing accelerated dynamics...")
    state = {"temperature": 300, "energy": 10.0}
    result = bridge.simulate_accelerated(
        target_time=1.0,
        scale=TimeScale.HOUR,
        state=state,
        acceleration_method="parallel_replica",
        parameters={"num_replicas": 8}
    )

    print(f"    Simulation completed:")
    print(f"      Simulation time: {result['simulation_time']:.2f}s")
    print(f"      Speedup: {result['speedup']}x")
    print(f"      Wall time: {result['wall_time']:.3f}s")


def example_5_orchestrator_aerogel():
    """Example 5: Multi-physics aerogel experiment"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Orchestrator - Multi-Department Aerogel Experiment")
    print("=" * 80)

    orchestrator = Orchestrator()

    # Create aerogel experiment
    experiment = orchestrator.create_aerogel_experiment()

    print(f"\nExperiment: {experiment.name}")
    print(f"Description: {experiment.description}")
    print(f"Departments involved: {', '.join(experiment.departments)}")
    print(f"Expected duration: {experiment.expected_duration}s")

    print(f"\nWorkflow structure:")
    print(f"  Total nodes: {len(experiment.workflow)}")
    print(f"  Total edges: {len(experiment.edges)}")

    print(f"\nWorkflow nodes:")
    for node_id, node in experiment.workflow.items():
        deps = f" (depends on: {', '.join(node.dependencies)})" if node.dependencies else ""
        print(f"  [{node_id}] {node.description}{deps}")

    print(f"\nExecution order would be:")
    # Simplified topological sort
    completed = set()
    level = 1
    while len(completed) < len(experiment.workflow):
        ready = [nid for nid, node in experiment.workflow.items()
                if nid not in completed and all(dep in completed for dep in node.dependencies)]
        if not ready:
            break
        print(f"  Level {level}: {', '.join(ready)}")
        completed.update(ready)
        level += 1


def example_6_level6_agents():
    """Example 6: Level-6 autonomous agents with learning"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Level-6 Autonomous Agents with Meta-Learning")
    print("=" * 80)

    async def run():
        # Create Level-6 agents
        physics_agent = create_level6_agent(AgentType.PHYSICS, "physics-advanced-001")
        materials_agent = create_level6_agent(AgentType.MATERIALS, "materials-advanced-001")

        print("\nCreated Level-6 agents:")
        print(f"  Physics Agent: {physics_agent.agent.agent_id}")
        print(f"    Capabilities: {', '.join(physics_agent.agent.capabilities)}")
        print(f"  Materials Agent: {materials_agent.agent.agent_id}")
        print(f"    Capabilities: {', '.join(materials_agent.agent.capabilities)}")

        # Test autonomous experiment proposal
        print("\n" + "-" * 80)
        print("Materials Agent: Autonomous Experiment Proposal")
        print("-" * 80)

        proposal = materials_agent.propose_experiment(
            "Find lightweight corrosion-resistant alloy for marine applications"
        )

        print(f"\nProposed experiment:")
        print(f"  Intent Type: {proposal['intent']['type']}")
        print(f"  Detected Materials: {proposal['intent']['materials']}")
        print(f"  Target Properties: {proposal['intent']['properties']}")
        print(f"  Design: {proposal['design']['num_runs']} runs using {proposal['design']['type']}")
        print(f"  Estimated Duration: {proposal['design']['estimated_duration']:.1f}s")
        print(f"  Resources: {proposal['resources']['cpu_cores']} CPU cores, {proposal['resources']['ram_mb']} MB RAM")

        # Execute tasks and learn
        print("\n" + "-" * 80)
        print("Physics Agent: Task Execution with Learning")
        print("-" * 80)

        tasks = [
            Task(
                task_id=f"physics_task_{i}",
                task_type="fluid_dynamics",
                description=f"Wind simulation {i}",
                priority=TaskPriority.NORMAL,
                required_capabilities=["fluid_dynamics"],
                parameters={"wind_speed": 20 + i * 5, "area": 2.0}
            )
            for i in range(5)
        ]

        print(f"\nExecuting {len(tasks)} tasks...")
        for task in tasks:
            result = await physics_agent.execute_task(task)
            print(f"  Task {task.task_id}: drag_force={result['drag_force']:.2f} N, confidence={result['confidence']:.2f}")

        # Self-evaluation
        print("\n" + "-" * 80)
        print("Agent Self-Evaluation")
        print("-" * 80)

        eval_physics = physics_agent.self_evaluate()
        print(f"\nPhysics Agent Performance:")
        print(f"  Average Performance: {eval_physics['avg_performance']:.3f}")
        print(f"  Learning Rate: {eval_physics['learning_rate']:.4f}")
        print(f"  Exploration Rate: {eval_physics['exploration_rate']:.3f}")
        print(f"  Knowledge Graph Size: {eval_physics['knowledge_nodes']} nodes")

        # Capabilities report
        print("\n" + "-" * 80)
        print("Agent Capabilities Report")
        print("-" * 80)

        report = physics_agent.get_capabilities_report()
        print(f"\nAgent: {report['agent_id']}")
        print(f"  Type: {report['agent_type']}")
        print(f"  Current Load: {report['current_load']:.2f}")
        print(f"  Tasks Completed: {report['performance_metrics']['tasks_completed']}")
        print(f"  Success Rate: {report['performance_metrics']['success_rate']:.2f}")
        print(f"  Knowledge Base: {report['knowledge_graph_size']} concepts")

    asyncio.run(run())


def example_7_complete_workflow():
    """Example 7: Complete workflow from intent to results"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Complete Workflow - Intent to Validated Results")
    print("=" * 80)

    print("\nStep 1: Parse user intent")
    print("-" * 40)
    query = "Optimize aerogel thermal insulation for cryogenic applications at -196°C"
    print(f"User query: '{query}'")

    parser = IntentParser()
    intent = parser.parse(query)
    print(f"  Parsed intent: {intent.experiment_type}")
    print(f"  Target properties: {intent.properties}")
    print(f"  Conditions: {intent.conditions}")

    print("\nStep 2: Design experiment")
    print("-" * 40)
    designer = ExperimentDesigner()
    design = designer.create_design(intent, ExperimentType.RESPONSE_SURFACE, num_runs=None)
    print(f"  Design type: {design.design_type.value}")
    print(f"  Number of runs: {design.num_runs}")
    print(f"  Parameters: {[p.name for p in design.parameters]}")

    print("\nStep 3: Estimate resources")
    print("-" * 40)
    estimator = ResourceEstimator()
    resources = estimator.estimate(design)
    print(f"  CPU cores: {resources['cpu_cores']}")
    print(f"  RAM required: {resources['ram_mb']} MB")
    print(f"  Estimated time: {resources['eta_hours']:.2f} hours")

    print("\nStep 4: Risk assessment")
    print("-" * 40)
    from hive_mind.crystalline_intent import RiskAssessment
    risk_assessor = RiskAssessment()
    risks = risk_assessor.assess(design)
    print(f"  Risk level: {risks['risk_level']}")
    print(f"  Number of risks: {risks['num_risks']}")
    print(f"  Recommended action: {risks['recommended_action']}")
    if risks['fallback_plans']:
        print(f"  Fallback plans:")
        for plan in risks['fallback_plans']:
            print(f"    - {plan}")

    print("\nStep 5: Knowledge graph query for similar experiments")
    print("-" * 40)
    kg = KnowledgeGraph()
    # Would query existing experiments here
    print(f"  Would query knowledge graph for similar aerogel experiments")
    print(f"  Would retrieve best practices and known failure modes")

    print("\nStep 6: Multi-agent execution (simulation)")
    print("-" * 40)
    print(f"  Would distribute {design.num_runs} runs across agent pool")
    print(f"  Materials agent: Load aerogel properties")
    print(f"  Environment agent: Set cryogenic conditions (-196°C)")
    print(f"  Physics agent: Calculate thermal conductivity")
    print(f"  Validation agent: Cross-check results")

    print("\nStep 7: Results aggregation and validation")
    print("-" * 40)
    print(f"  Would aggregate results from all agents")
    print(f"  Would validate energy conservation")
    print(f"  Would check consistency across departments")
    print(f"  Would generate confidence intervals")

    print("\nWorkflow complete!")
    print(f"Total estimated time: {resources['eta_hours']:.2f} hours")
    print(f"Expected confidence: >0.85")


def run_all_examples():
    """Run all examples"""
    examples = [
        example_1_basic_hive_mind,
        example_2_knowledge_graph,
        example_3_crystalline_intent,
        example_4_temporal_bridge,
        example_5_orchestrator_aerogel,
        example_6_level6_agents,
        example_7_complete_workflow
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_examples()
