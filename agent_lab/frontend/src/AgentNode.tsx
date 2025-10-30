import React, { memo } from 'react';
import { Handle, Position } from 'reactflow';

const AgentNode = ({ data, id, onDataChange }) => {
  const onTextChange = (event) => {
    if (onDataChange) {
      onDataChange(id, { ...data, parameters: event.target.value });
    }
  };

  return (
    <div style={{
      padding: '10px',
      border: '1a192b',
      borderRadius: '5px',
      background: '#fff',
      width: '200px',
    }}>
      <Handle type="target" position={Position.Top} />
      <div>
        <strong>{data.label}</strong>
      </div>
      <div style={{ marginTop: '10px' }}>
        <label htmlFor="text">Task Parameters:</label>
        <textarea
          id="text"
          name="text"
          rows={4}
          style={{ width: '100%' }}
          defaultValue={data.parameters || ''}
          onChange={onTextChange}
        />
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};

export default memo(AgentNode);
