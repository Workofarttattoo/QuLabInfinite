import { useEffect, useState } from "react";
import CircuitPane from "../widgets/CircuitPane";
import FidelityChart from "../widgets/FidelityChart";

type Result = { fidelity: number; counts: Record<string, number>; shots: number };

export default function Teleport() {
  const [status, setStatus] = useState<string>("idle");
  const [result, setResult] = useState<Result| null>(null);
  const [running, setRunning] = useState(false);

  const run = () => {
    setRunning(true);
    setStatus("connecting");
    setResult(null);
    const evt = new EventSource("http://localhost:8000/sim/teleport/run");
    evt.onmessage = (e) => {
      setStatus("message");
      // SSE default event
    };
    evt.addEventListener("status", (e:any) => setStatus((e as MessageEvent).data));
    evt.addEventListener("result", (e:any) => {
      const r = JSON.parse((e as MessageEvent).data);
      setResult(r); setRunning(false); evt.close();
    });
    evt.onerror = () => { setStatus("error"); setRunning(false); evt.close(); };
  };

  return (
    <div className="card">
      <h1>Teleportation Workspace (Lite)</h1>
      <p>Build a circuit, then run a stubbed simulation (Pro/Lab calls Qiskit).</p>
      <CircuitPane/>
      <div style={{marginTop:12, display:"flex", gap:8}}>
        <button className="btn" disabled={running} onClick={run}>
          {running ? "Running..." : "Run (SSE)"}
        </button>
        <span>Status: {status}</span>
      </div>
      {result && (
        <>
          <h3 style={{marginTop:16}}>Result</h3>
          <div>Fidelity: {result.fidelity}</div>
          <FidelityChart counts={result.counts}/>
        </>
      )}
    </div>
  );
}