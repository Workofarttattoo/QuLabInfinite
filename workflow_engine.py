"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

QULAB WORKFLOW ENGINE
Multi-lab orchestration and data pipeline system
"""

import asyncio
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import uuid

@dataclass
class WorkflowNode:
    """Single node in workflow graph"""
    id: str
    lab_name: str
    parameters: Dict[str, Any]
    inputs: List[str] = None  # IDs of input nodes
    outputs: List[str] = None  # IDs of output nodes
    result: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, running, completed, failed
    execution_time: float = 0.0

@dataclass
class DataTransform:
    """Transform data between lab formats"""
    source_lab: str
    target_lab: str
    transform_fn: Callable

class WorkflowEngine:
    """
    Orchestrates multi-lab workflows with:
    - DAG execution
    - Data format conversion
    - Parallel processing
    - Error recovery
    - Visual workflow design
    """

    def __init__(self):
        self.workflows = {}
        self.transforms = {}
        self.execution_history = []
        self._init_transforms()

    def _init_transforms(self):
        """Initialize data format transformers between labs"""

        # Tumor → Drug transforms
        self.register_transform(
            "tumor_growth_simulator",
            "drug_resistance_evolution",
            lambda data: {
                "tumor_volume": data.get("tumor_volume", 100),
                "growth_rate": data.get("growth_rate", 0.3),
                "mutation_rate": 1e-6
            }
        )

        # Drug screening → Clinical
        self.register_transform(
            "virtual_screening",
            "clinical_trial_simulator",
            lambda data: {
                "drug_candidates": data.get("hits", []),
                "efficacy": data.get("binding_affinity", -8),
                "patient_count": 1000
            }
        )

        # Protein → Drug
        self.register_transform(
            "protein_folding_simulator",
            "molecular_docking",
            lambda data: {
                "receptor_structure": data.get("structure", {}),
                "binding_site": data.get("active_site", {}),
                "grid_size": 20
            }
        )

        # Genomics → Clinical
        self.register_transform(
            "mutation_impact_predictor",
            "treatment_outcome_predictor",
            lambda data: {
                "mutations": data.get("pathogenic_variants", []),
                "risk_score": data.get("impact_score", 0.5),
                "treatment_type": "targeted"
            }
        )

    def register_transform(self, source_lab: str, target_lab: str, transform_fn: Callable):
        """Register a data transformer between labs"""
        key = f"{source_lab}→{target_lab}"
        self.transforms[key] = DataTransform(source_lab, target_lab, transform_fn)

    def create_workflow(self, name: str) -> str:
        """Create a new workflow"""
        workflow_id = str(uuid.uuid4())
        self.workflows[workflow_id] = {
            "name": name,
            "nodes": {},
            "graph": nx.DiGraph(),
            "created_at": datetime.now().isoformat(),
            "status": "draft"
        }
        return workflow_id

    def add_node(self, workflow_id: str, lab_name: str, parameters: Dict[str, Any]) -> str:
        """Add a node to the workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        node_id = f"{lab_name}_{len(self.workflows[workflow_id]['nodes'])}"
        node = WorkflowNode(
            id=node_id,
            lab_name=lab_name,
            parameters=parameters,
            inputs=[],
            outputs=[]
        )

        self.workflows[workflow_id]["nodes"][node_id] = node
        self.workflows[workflow_id]["graph"].add_node(node_id, lab=lab_name)

        return node_id

    def connect_nodes(self, workflow_id: str, source_id: str, target_id: str):
        """Connect two nodes in the workflow"""
        workflow = self.workflows[workflow_id]

        if source_id not in workflow["nodes"] or target_id not in workflow["nodes"]:
            raise ValueError("Invalid node IDs")

        # Update node connections
        workflow["nodes"][source_id].outputs.append(target_id)
        workflow["nodes"][target_id].inputs.append(source_id)

        # Update graph
        workflow["graph"].add_edge(source_id, target_id)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(workflow["graph"]):
            # Revert changes
            workflow["nodes"][source_id].outputs.remove(target_id)
            workflow["nodes"][target_id].inputs.remove(source_id)
            workflow["graph"].remove_edge(source_id, target_id)
            raise ValueError("Connection would create a cycle!")

    def validate_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Validate workflow is ready for execution"""
        workflow = self.workflows[workflow_id]
        graph = workflow["graph"]

        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Check if graph is empty
        if len(graph.nodes) == 0:
            validation["valid"] = False
            validation["errors"].append("Workflow has no nodes")
            return validation

        # Check for cycles
        if not nx.is_directed_acyclic_graph(graph):
            validation["valid"] = False
            validation["errors"].append("Workflow contains cycles")

        # Check for disconnected components
        if not nx.is_weakly_connected(graph):
            validation["warnings"].append("Workflow has disconnected components")

        # Check data format compatibility
        for edge in graph.edges():
            source_node = workflow["nodes"][edge[0]]
            target_node = workflow["nodes"][edge[1]]
            transform_key = f"{source_node.lab_name}→{target_node.lab_name}"

            if transform_key not in self.transforms:
                validation["warnings"].append(
                    f"No transform defined for {source_node.lab_name} → {target_node.lab_name}"
                )

        return validation

    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute the workflow"""
        workflow = self.workflows[workflow_id]

        # Validate first
        validation = self.validate_workflow(workflow_id)
        if not validation["valid"]:
            return {
                "status": "failed",
                "errors": validation["errors"]
            }

        workflow["status"] = "running"
        workflow["started_at"] = datetime.now().isoformat()

        try:
            # Get execution order (topological sort)
            execution_order = list(nx.topological_sort(workflow["graph"]))

            # Execute nodes
            for node_id in execution_order:
                await self._execute_node(workflow_id, node_id)

            workflow["status"] = "completed"
            workflow["completed_at"] = datetime.now().isoformat()

            # Collect results
            results = {
                node_id: node.result
                for node_id, node in workflow["nodes"].items()
                if node.result is not None
            }

            # Save to history
            self.execution_history.append({
                "workflow_id": workflow_id,
                "name": workflow["name"],
                "timestamp": datetime.now().isoformat(),
                "results": results
            })

            return {
                "status": "completed",
                "results": results,
                "execution_time": sum(n.execution_time for n in workflow["nodes"].values())
            }

        except Exception as e:
            workflow["status"] = "failed"
            workflow["error"] = str(e)
            return {
                "status": "failed",
                "error": str(e)
            }

    async def _execute_node(self, workflow_id: str, node_id: str):
        """Execute a single node in the workflow"""
        workflow = self.workflows[workflow_id]
        node = workflow["nodes"][node_id]

        node.status = "running"
        start_time = asyncio.get_event_loop().time()

        try:
            # Collect inputs from predecessor nodes
            input_data = {}
            for input_id in node.inputs or []:
                input_node = workflow["nodes"][input_id]
                if input_node.result:
                    # Apply transform if available
                    transform_key = f"{input_node.lab_name}→{node.lab_name}"
                    if transform_key in self.transforms:
                        transform = self.transforms[transform_key]
                        input_data[input_id] = transform.transform_fn(input_node.result)
                    else:
                        input_data[input_id] = input_node.result

            # Merge input data with parameters
            execution_params = node.parameters.copy()
            if input_data:
                execution_params["input_data"] = input_data

            # Simulate lab execution (would call actual API in production)
            result = await self._simulate_lab_execution(node.lab_name, execution_params)

            node.result = result
            node.status = "completed"
            node.execution_time = asyncio.get_event_loop().time() - start_time

        except Exception as e:
            node.status = "failed"
            node.error = str(e)
            raise

    async def _simulate_lab_execution(self, lab_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate lab execution (placeholder for actual API call)"""
        # Simulate computation delay
        await asyncio.sleep(np.random.uniform(0.5, 2))

        # Return simulated results
        return {
            "lab": lab_name,
            "timestamp": datetime.now().isoformat(),
            "value": np.random.randn(),
            "parameters": parameters
        }

    def visualize_workflow(self, workflow_id: str, output_file: Optional[str] = None):
        """Create visual representation of workflow"""
        workflow = self.workflows[workflow_id]
        graph = workflow["graph"]

        plt.figure(figsize=(12, 8))

        # Create layout
        pos = nx.spring_layout(graph, seed=42, k=2)

        # Draw nodes with status colors
        node_colors = []
        for node_id in graph.nodes():
            node = workflow["nodes"][node_id]
            if node.status == "completed":
                node_colors.append("green")
            elif node.status == "running":
                node_colors.append("yellow")
            elif node.status == "failed":
                node_colors.append("red")
            else:
                node_colors.append("lightgray")

        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=1000,
            alpha=0.8
        )

        # Draw edges
        nx.draw_networkx_edges(
            graph, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            alpha=0.5
        )

        # Draw labels
        labels = {
            node_id: workflow["nodes"][node_id].lab_name.replace("_", "\n")
            for node_id in graph.nodes()
        }
        nx.draw_networkx_labels(
            graph, pos,
            labels,
            font_size=8
        )

        plt.title(f"Workflow: {workflow['name']}")
        plt.axis("off")

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def export_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Export workflow as JSON"""
        workflow = self.workflows[workflow_id]

        return {
            "name": workflow["name"],
            "created_at": workflow["created_at"],
            "nodes": [
                {
                    "id": node.id,
                    "lab": node.lab_name,
                    "parameters": node.parameters,
                    "inputs": node.inputs,
                    "outputs": node.outputs
                }
                for node in workflow["nodes"].values()
            ],
            "edges": list(workflow["graph"].edges())
        }

    def import_workflow(self, workflow_data: Dict[str, Any]) -> str:
        """Import workflow from JSON"""
        workflow_id = self.create_workflow(workflow_data["name"])

        # Add nodes
        node_mapping = {}
        for node_data in workflow_data["nodes"]:
            new_node_id = self.add_node(
                workflow_id,
                node_data["lab"],
                node_data["parameters"]
            )
            node_mapping[node_data["id"]] = new_node_id

        # Add edges
        for source, target in workflow_data["edges"]:
            self.connect_nodes(
                workflow_id,
                node_mapping[source],
                node_mapping[target]
            )

        return workflow_id

# =============================================================================
# PREDEFINED WORKFLOWS
# =============================================================================

def create_drug_discovery_workflow(engine: WorkflowEngine) -> str:
    """Create a complete drug discovery workflow"""
    wf_id = engine.create_workflow("Drug Discovery Pipeline")

    # Add nodes
    target = engine.add_node(wf_id, "target_identification", {"disease": "cancer"})
    screen = engine.add_node(wf_id, "virtual_screening", {"library_size": 10000})
    dock = engine.add_node(wf_id, "molecular_docking", {"precision": "high"})
    admet = engine.add_node(wf_id, "admet_predictor", {"model": "advanced"})
    tox = engine.add_node(wf_id, "toxicity_predictor", {"assays": 5})
    opt = engine.add_node(wf_id, "lead_optimization", {"iterations": 10})

    # Connect nodes
    engine.connect_nodes(wf_id, target, screen)
    engine.connect_nodes(wf_id, screen, dock)
    engine.connect_nodes(wf_id, dock, admet)
    engine.connect_nodes(wf_id, admet, tox)
    engine.connect_nodes(wf_id, tox, opt)

    return wf_id

def create_personalized_cancer_workflow(engine: WorkflowEngine) -> str:
    """Create personalized cancer treatment workflow"""
    wf_id = engine.create_workflow("Personalized Cancer Treatment")

    # Add nodes
    tumor = engine.add_node(wf_id, "tumor_growth_simulator", {"type": "lung"})
    mutation = engine.add_node(wf_id, "mutation_impact_predictor", {"panel": "comprehensive"})
    microenv = engine.add_node(wf_id, "tumor_microenvironment", {"immune": True})
    resistance = engine.add_node(wf_id, "drug_resistance_evolution", {"timepoints": 10})
    chemo = engine.add_node(wf_id, "chemotherapy_optimizer", {"drugs": ["cisplatin", "docetaxel"]})
    immuno = engine.add_node(wf_id, "immunotherapy_response", {"checkpoint": "PD-1"})
    outcome = engine.add_node(wf_id, "treatment_outcome_predictor", {"followup_months": 24})

    # Connect nodes
    engine.connect_nodes(wf_id, tumor, mutation)
    engine.connect_nodes(wf_id, tumor, microenv)
    engine.connect_nodes(wf_id, mutation, resistance)
    engine.connect_nodes(wf_id, microenv, immuno)
    engine.connect_nodes(wf_id, resistance, chemo)
    engine.connect_nodes(wf_id, chemo, outcome)
    engine.connect_nodes(wf_id, immuno, outcome)

    return wf_id

def create_protein_engineering_workflow(engine: WorkflowEngine) -> str:
    """Create protein engineering workflow"""
    wf_id = engine.create_workflow("Protein Engineering")

    # Add nodes
    fold = engine.add_node(wf_id, "protein_folding_simulator", {"sequence": "MKTLLIFLSL"})
    align = engine.add_node(wf_id, "structural_alignment", {"reference": "1ABC"})
    epitope = engine.add_node(wf_id, "epitope_predictor", {"mhc_class": "I"})
    interact = engine.add_node(wf_id, "protein_protein_interaction", {"partners": ["CD4", "CD8"]})
    dock = engine.add_node(wf_id, "molecular_docking", {"ligand": "inhibitor"})

    # Connect nodes
    engine.connect_nodes(wf_id, fold, align)
    engine.connect_nodes(wf_id, align, epitope)
    engine.connect_nodes(wf_id, epitope, interact)
    engine.connect_nodes(wf_id, fold, dock)

    return wf_id

# =============================================================================
# BATCH PROCESSING
# =============================================================================

class BatchProcessor:
    """Process multiple workflows or datasets in batch"""

    def __init__(self, engine: WorkflowEngine):
        self.engine = engine
        self.batch_results = []

    async def process_batch(self, workflow_id: str, datasets: List[Dict[str, Any]]):
        """Process multiple datasets through same workflow"""
        results = []

        for i, dataset in enumerate(datasets):
            print(f"Processing dataset {i+1}/{len(datasets)}")

            # Update workflow parameters with dataset
            workflow = self.engine.workflows[workflow_id]
            for node in workflow["nodes"].values():
                node.parameters.update(dataset)
                node.result = None  # Reset results
                node.status = "pending"

            # Execute workflow
            result = await self.engine.execute_workflow(workflow_id)
            results.append(result)

        self.batch_results = results
        return results

    def aggregate_results(self) -> Dict[str, Any]:
        """Aggregate batch results"""
        if not self.batch_results:
            return {}

        # Collect all numeric values
        aggregated = {
            "batch_size": len(self.batch_results),
            "successful": sum(1 for r in self.batch_results if r["status"] == "completed"),
            "failed": sum(1 for r in self.batch_results if r["status"] == "failed"),
            "mean_execution_time": np.mean([
                r.get("execution_time", 0) for r in self.batch_results
            ])
        }

        return aggregated

# =============================================================================
# MAIN DEMO
# =============================================================================

async def main():
    """Demo the workflow engine"""
    print("=" * 80)
    print("QULAB WORKFLOW ENGINE DEMO")
    print("=" * 80)

    engine = WorkflowEngine()

    # Create workflows
    drug_wf = create_drug_discovery_workflow(engine)
    cancer_wf = create_personalized_cancer_workflow(engine)
    protein_wf = create_protein_engineering_workflow(engine)

    print(f"\nCreated 3 workflows:")
    print(f"1. Drug Discovery: {drug_wf}")
    print(f"2. Cancer Treatment: {cancer_wf}")
    print(f"3. Protein Engineering: {protein_wf}")

    # Validate workflows
    for wf_id in [drug_wf, cancer_wf, protein_wf]:
        validation = engine.validate_workflow(wf_id)
        name = engine.workflows[wf_id]["name"]
        print(f"\n{name}: Valid={validation['valid']}")
        if validation["warnings"]:
            print(f"  Warnings: {validation['warnings']}")

    # Execute drug discovery workflow
    print("\n" + "=" * 40)
    print("Executing Drug Discovery Workflow...")
    print("=" * 40)

    result = await engine.execute_workflow(drug_wf)
    print(f"Status: {result['status']}")
    print(f"Execution time: {result.get('execution_time', 0):.2f}s")

    # Visualize workflow
    engine.visualize_workflow(drug_wf, "drug_workflow.png")
    print("\n✅ Workflow visualization saved to drug_workflow.png")

    # Export workflow
    export_data = engine.export_workflow(drug_wf)
    with open("drug_workflow.json", "w") as f:
        json.dump(export_data, f, indent=2)
    print("✅ Workflow exported to drug_workflow.json")

    # Batch processing demo
    print("\n" + "=" * 40)
    print("Batch Processing Demo")
    print("=" * 40)

    batch = BatchProcessor(engine)
    datasets = [
        {"disease": "lung_cancer", "library_size": 5000},
        {"disease": "breast_cancer", "library_size": 8000},
        {"disease": "prostate_cancer", "library_size": 10000}
    ]

    batch_results = await batch.process_batch(drug_wf, datasets)
    aggregated = batch.aggregate_results()

    print(f"\nBatch Results:")
    print(f"  Processed: {aggregated['batch_size']} datasets")
    print(f"  Successful: {aggregated['successful']}")
    print(f"  Failed: {aggregated['failed']}")
    print(f"  Mean time: {aggregated['mean_execution_time']:.2f}s")

    print("\n✅ Workflow engine demo complete!")

if __name__ == "__main__":
    asyncio.run(main())