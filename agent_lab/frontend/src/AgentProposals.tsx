import React, { useEffect, useState } from 'react';

const AgentProposals = ({ newProposal }) => {
  const [proposals, setProposals] = useState([]);

  useEffect(() => {
    if (newProposal) {
      setProposals(prev => [newProposal, ...prev]);
    }
  }, [newProposal]);

  return (
    <div style={{ padding: '10px', borderTop: '1px solid #ddd' }}>
      <h4>Agent Proposals</h4>
      {proposals.length === 0 && <p>No proposals yet.</p>}
      {proposals.map((p, i) => (
        <div key={i} style={{ marginBottom: '10px', padding: '5px', border: '1px solid #eee' }}>
          <strong>Proposal from {p.agent_id}</strong>
          <p><strong>Goal:</strong> {p.proposal.intent.type}</p>
          <p><strong>Num Runs:</strong> {p.proposal.design.num_runs}</p>
          <button onClick={() => alert("Executing this proposal is not yet implemented.")}>
            Approve & Run
          </button>
        </div>
      ))}
    </div>
  );
};

export default AgentProposals;
