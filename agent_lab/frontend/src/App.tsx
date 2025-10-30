import React, { useState, useCallback, useRef, useEffect } from 'react';
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  addEdge,
  useNodesState,
  useEdgesState,
  ReactFlowProvider,
} from 'reactflow';
import 'reactflow/dist/style.css';
import Sidebar from './Sidebar';
import axios from 'axios';
import AgentNode from './AgentNode';
import AgentFeed from './AgentFeed';
import VoiceControl from './VoiceControl';
import AgentProposals from './AgentProposals';

const nodeTypes = {
  agentNode: (props) => <AgentNode {...props} onDataChange={onNodeDataChange} />,
};

const initialNodes = [
  { id: '1', type: 'input', position: { x: 250, y: 5 }, data: { label: 'Start' } },
];

const initialEdges = [];

let id = 2;
const getId = () => `${id++}`;

function App() {
  const reactFlowWrapper = useRef(null);
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [reactFlowInstance, setReactFlowInstance] = useState(null);
  const [latestProposal, setLatestProposal] = useState(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/agent-feed');
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'agent_proposal') {
        setLatestProposal(message);
      }
    };
    return () => ws.close();
  }, []);

  const onNodeDataChange = useCallback((id, data) => {
    setNodes((nds) =>
      nds.map((node) => (node.id === id ? { ...node, data } : node))
    );
  }, [setNodes]);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  );

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow');
      const agentData = JSON.parse(event.dataTransfer.getData('application/json'));

      if (typeof type === 'undefined' || !type) {
        return;
      }

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      const newNode = {
        id: getId(),
        type: 'agentNode',
        position,
        data: {
          label: `${agentData.agent_type} Node`,
          agent: agentData,
          parameters: '',
          onDataChange: onNodeDataChange,
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, onNodeDataChange],
  );

  const onRunWorkflow = useCallback(() => {
    const workflow = {
      nodes: nodes.map(node => ({
        id: node.id,
        type: node.type,
        position: node.position,
        data: node.data,
      })),
      edges: edges.map(edge => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
      })),
    };

    axios.post('http://localhost:8000/workflows', workflow)
      .then(response => {
        alert('Workflow executed successfully!');
        console.log('Workflow execution result:', response.data);
      })
      .catch(error => {
        alert('Failed to execute workflow.');
        console.error('Error executing workflow:', error);
      });
  }, [nodes, edges]);

  const memoizedNodeTypes = React.useMemo(() => ({
    agentNode: (props) => <AgentNode {...props} onDataChange={onNodeDataChange} />,
  }), [onNodeDataChange]);

  return (
    <div className="app-container" ref={reactFlowWrapper}>
      <ReactFlowProvider>
        <Sidebar />
        <div className="main-content">
          <div className="reactflow-wrapper">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onInit={setReactFlowInstance}
              onDrop={onDrop}
              onDragOver={onDragOver}
              fitView
              nodeTypes={memoizedNodeTypes}
            >
              <Controls />
              <MiniMap />
              <Background variant="dots" gap={12} size={1} />
            </ReactFlow>
          </div>
          <AgentFeed />
        </div>
        <div className="sidebar-right">
            <VoiceControl />
            <AgentProposals newProposal={latestProposal} />
        </div>
        <div className="run-button-container">
          <button onClick={onRunWorkflow}>Run Workflow</button>
        </div>
      </ReactFlowProvider>
    </div>
  );
}

export default App;
