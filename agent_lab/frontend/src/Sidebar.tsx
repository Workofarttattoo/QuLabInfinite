import React, { useEffect, useState } from 'react';
import axios from 'axios';

const Sidebar = () => {
  const [agents, setAgents] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8000/agents')
      .then(response => {
        setAgents(response.data);
      })
      .catch(error => {
        console.error('Error fetching agents:', error);
      });
  }, []);

  const onDragStart = (event: React.DragEvent, nodeType: string, agent: any) => {
    const agentData = JSON.stringify(agent);
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.setData('application/json', agentData);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <aside style={{ width: '250px', borderRight: '1px solid #ddd', padding: '10px' }}>
      <h2>Agents</h2>
      {agents.length > 0 ? (
        <div>
          {agents.map((agent: any) => (
            <div
              key={agent.agent_id}
              onDragStart={(event) => onDragStart(event, 'default', agent)}
              draggable
              style={{
                padding: '10px',
                border: '1px solid #ccc',
                borderRadius: '5px',
                marginBottom: '10px',
                cursor: 'grab',
              }}
            >
              <strong>{agent.agent_type}</strong>
              <p>Capabilities: {agent.capabilities.join(', ')}</p>
            </div>
          ))}
        </div>
      ) : (
        <p>Loading agents...</p>
      )}
    </aside>
  );
};

export default Sidebar;
