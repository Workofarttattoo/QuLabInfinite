"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

AI Agent Lab - Backend Service
Exposes Hive Mind functionality via a REST API for the agent workflow UI.
"""
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Callable, Set
import uuid
from fastapi.websockets import WebSocketDisconnect
import json
import time

# Assuming the QuLabInfinite project is in the Python path
from hive_mind.hive_mind_core import HiveMind, create_standard_agents, TaskPriority
from hive_mind.orchestrator import Orchestrator, MultiPhysicsExperiment, WorkflowNode, WorkflowEdge
from api.ech0_bridge import ECH0Bridge

app = FastAPI(
    title="AI Agent Lab API",
    description="API for interacting with the Hive Mind and orchestrating agentic workflows.",
    version="0.1.0",
)

orchestrator: Orchestrator = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/agent-feed")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    """Initialize the Hive Mind and Orchestrator on startup."""
    global orchestrator
    orchestrator = Orchestrator()
    
    async def proposal_callback(agent_id: str, proposal: Dict[str, Any]):
        """Callback to send proposals to the frontend."""
        await manager.broadcast(json.dumps({
            "type": "agent_proposal",
            "agent_id": agent_id,
            "proposal": proposal,
            "timestamp": time.time(),
        }))

    # Pass the broadcast function to the hive_mind
    orchestrator.hive_mind.set_broadcast_callback(manager.broadcast)
    orchestrator.hive_mind.set_proposal_callback(proposal_callback)
    
    await orchestrator.initialize()
    # Register standard agents by passing the hive_mind instance
    create_standard_agents(orchestrator.hive_mind)
    
    # --- ECH0 Integration ---
    # Create an instance of the bridge and subscribe ech0 to the hearing channel
    ech0_bridge = ECH0Bridge()
    ech0_bridge.hive_mind = orchestrator.hive_mind # Ensure the bridge uses the main hive_mind instance
    
    def ech0_callback(data: Dict[str, Any]):
        """This is where you would send the data to the ECH0 LLM."""
        print(f"[ECH0 HEARD]: {data.get('text')}")
        # In a real implementation, you might send this over a network socket,
        # write to a database, or call an external API.
    
    ech0_bridge.subscribe_to_hearing_channel(ech0_callback)
    # ----------------------

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the Orchestrator on shutdown."""
    if orchestrator:
        await orchestrator.shutdown()

@app.get("/agents", summary="Get a list of available agents")
def get_agents() -> List[Dict[str, Any]]:
    """
    Retrieves a list of all registered agents in the Hive Mind,
    including their capabilities and current status.
    """
    agents = orchestrator.hive_mind.registry.agents.values()
    return [
        {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type.value,
            "capabilities": agent.capabilities,
            "status": agent.status,
            "current_load": agent.current_load,
        }
        for agent in agents
    ]

class NodeData(BaseModel):
    id: str
    type: str
    position: Dict[str, float]
    data: Dict[str, Any]

class EdgeData(BaseModel):
    id: str
    source: str
    target: str

class WorkflowPayload(BaseModel):
    nodes: List[NodeData]
    edges: List[EdgeData]

class HearingPayload(BaseModel):
    text: str

class ProposalPayload(BaseModel):
    agent_id: str
    proposal: Dict[str, Any]

@app.post("/broadcast/hearing", summary="Broadcast a message to all agents")
async def broadcast_hearing(payload: HearingPayload):
    """
    Receives a text message and publishes it to the 'hearing_channel'
    in the hive_mind's knowledge base, making it available to all agents.
    """
    orchestrator.hive_mind.knowledge.publish(
        topic="hearing_channel",
        data={"text": payload.text, "source": "human_operator"},
        source_agent="human_operator"
    )
    return {"status": "broadcasted", "text": payload.text}

@app.post("/workflows", summary="Create and execute a new workflow")
async def create_and_execute_workflow(payload: WorkflowPayload):
    """
    Receives a workflow definition from the frontend, translates it into a
    MultiPhysicsExperiment, and executes it.
    """
    workflow_nodes = {}
    edges = []

    for node_data in payload.nodes:
        # We don't create hive_mind nodes for the UI 'input' node
        if node_data.type == 'input' or node_data.type == 'start':
            continue

        agent_info = node_data.data.get('agent', {})
        parameters = node_data.data.get('parameters', '')
        
        # Simple parsing of parameters from string to dict
        # Assumes format: key=value, key2=value2
        try:
            param_dict = dict(item.split("=") for item in parameters.split(",") if item)
        except ValueError:
            param_dict = {"raw_text": parameters}


        task_spec = {
            "type": agent_info.get('agent_type'),
            "capabilities": agent_info.get('capabilities', []),
            "parameters": param_dict,
            "duration": 60.0  # Default duration
        }
        
        workflow_nodes[node_data.id] = WorkflowNode(
            node_id=node_data.id,
            node_type="task",
            description=node_data.data.get('label', 'No description'),
            task_spec=task_spec,
            dependencies=[]
        )

    for edge_data in payload.edges:
        source_id = edge_data.source
        target_id = edge_data.target
        
        # Add dependency to the target node
        if target_id in workflow_nodes:
            # Check if the source is not the start node
            source_node_is_start = any(n.id == source_id and n.type == 'input' for n in payload.nodes)
            if not source_node_is_start:
                 workflow_nodes[target_id].dependencies.append(source_id)

        edges.append(WorkflowEdge(source=source_id, target=target_id))


    experiment = MultiPhysicsExperiment(
        experiment_id=f"dynamic_exp_{uuid.uuid4()}",
        name="Dynamically Generated Workflow",
        description="Workflow created from the Agent Lab UI",
        departments=list(set(node.task_spec.get('type') for node in workflow_nodes.values() if node.task_spec)),
        workflow=workflow_nodes,
        edges=edges,
        parameters={},
        expected_duration=len(workflow_nodes) * 60.0,
        priority=TaskPriority.HIGH
    )

    results = await orchestrator.execute_experiment(experiment)
    return results

@app.get("/workflows/{workflow_id}", summary="Get the status of a workflow")
def get_workflow_status(workflow_id: str):
    """
    Retrieves the current status and results of a specific workflow.
    """
    status = orchestrator.workflow_engine.workflow_status.get(workflow_id, "NOT_FOUND")
    results = orchestrator.workflow_engine.node_results.get(workflow_id, {})
    return {"workflow_id": workflow_id, "status": status, "results": results}

@app.get("/status", summary="Get the overall status of the Hive Mind")
def get_hive_mind_status():
    """
    Returns a status summary of the entire Hive Mind, including agent and task queue stats.
    """
    return orchestrator.hive_mind.get_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
