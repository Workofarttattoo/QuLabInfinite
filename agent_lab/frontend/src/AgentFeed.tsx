import React, { useEffect, useState } from 'react';

const AgentFeed = () => {
  const [messages, setMessages] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/agent-feed');

    ws.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setMessages((prevMessages) => [message, ...prevMessages]);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div style={{
      width: '100%',
      height: '200px',
      borderTop: '1px solid #ddd',
      padding: '10px',
      overflowY: 'scroll',
      display: 'flex',
      flexDirection: 'column-reverse',
    }}>
      <h3>Agent Activity Feed ({isConnected ? 'Connected' : 'Disconnected'})</h3>
      <div>
        {messages.map((msg, index) => (
          <div key={index} style={{ marginBottom: '5px', padding: '5px', border: '1px solid #eee' }}>
            <strong>{msg.type}</strong>
            <p style={{ margin: 0 }}>{msg.description || `Task ${msg.task_id} ${msg.status}`}</p>
            <small>{new Date(msg.timestamp * 1000).toLocaleTimeString()}</small>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentFeed;
