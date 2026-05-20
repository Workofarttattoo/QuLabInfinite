import { Navigation } from '../components/Navigation';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function QuantumLab() {
  const { health, loading } = useLabHealth('quantum');

  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 px-[32px]">
          <EchoLabCommandInline className="mb-8" />
        <div className="max-w-[1440px] mx-auto">
          <div className="mb-12">
            <div className="mb-4 inline-flex items-center gap-2 px-3 py-1 bg-[rgba(0,240,255,0.2)] border border-[rgba(0,240,255,0.4)] rounded-full">
              <span className="w-2 h-2 rounded-full bg-[#00dbe9] animate-pulse"></span>
              <span className="text-[12px] leading-[16px] tracking-[0.15em] font-bold text-[#00dbe9]">QUANTUM_COMPUTING_LAB</span>
            </div>
            <h1 className="text-[48px] leading-[56px] tracking-[-0.02em] font-bold text-[#00dbe9] mb-4">
              Quantum Computing Lab
            </h1>
            <p className="text-[18px] leading-[28px] text-[#b9cacb] max-w-3xl">
              Quantum circuit simulation with VQE, QAOA algorithms - Unified API Port 8000
            </p>
          </div>

          <div className="grid grid-cols-4 gap-6 mb-8">
            <div className="glass-panel p-6 rounded-xl neon-glow-cyan">
              <h3 className="text-[24px] font-semibold text-[#00dbe9] mb-4">Lab Status</h3>
              {loading ? (
                <div className="text-[#b9cacb]">Loading...</div>
              ) : health ? (
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-3 h-3 rounded-full bg-[#00dbe9] animate-pulse"></span>
                    <span className="text-[#00dbe9] font-semibold">✅ {health.status || 'ONLINE'}</span>
                  </div>
                  <div className="text-[12px] text-[#b9cacb] mt-4">Unified API Connected</div>
                </div>
              ) : (
                <div className="text-[#ffb4ab]">❌ Offline</div>
              )}
            </div>

            <div className="glass-panel p-6 rounded-xl border-white/10">
              <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-2">MAX QUBITS</div>
              <div className="text-[32px] font-bold text-[#00dbe9]">50</div>
              <div className="text-[12px] text-[#b9cacb] mt-2">Tensor network</div>
            </div>

            <div className="glass-panel p-6 rounded-xl border-white/10">
              <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-2">FIDELITY</div>
              <div className="text-[32px] font-bold text-[#ddb7ff]">0.99</div>
              <div className="text-[12px] text-[#b9cacb] mt-2">Statevector ≤30 qubits</div>
            </div>

            <div className="glass-panel p-6 rounded-xl border-white/10">
              <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-2">API ENDPOINT</div>
              <div className="text-[12px] text-[#00dbe9] font-mono">/quantum/simulate</div>
              <div className="text-[12px] text-[#b9cacb] mt-2">POST to Unified API</div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-6">
            <div className="glass-panel p-6 rounded-xl border-white/10">
              <h3 className="text-[20px] font-semibold text-[#00dbe9] mb-4">Algorithms</h3>
              <div className="space-y-2 text-[12px]">
                <div className="px-2 py-1 bg-[rgba(0,219,233,0.15)] rounded text-[#00dbe9]">VQE (Variational Quantum Eigensolver)</div>
                <div className="px-2 py-1 bg-[rgba(0,219,233,0.15)] rounded text-[#00dbe9]">QAOA (Quantum Approximation)</div>
                <div className="px-2 py-1 bg-[rgba(0,219,233,0.15)] rounded text-[#00dbe9]">Quantum Annealing</div>
              </div>
            </div>

            <div className="glass-panel p-6 rounded-xl border-white/10">
              <h3 className="text-[20px] font-semibold text-[#ddb7ff] mb-4">Gate Set</h3>
              <div className="flex flex-wrap gap-2">
                {['H', 'X', 'RX', 'RY', 'RZ', 'CNOT', 'CZ'].map(gate => (
                  <div key={gate} className="px-3 py-1 bg-[rgba(168,85,247,0.15)] rounded text-[#ddb7ff] text-[11px] font-mono">
                    {gate}
                  </div>
                ))}
              </div>
            </div>

            <div className="glass-panel p-6 rounded-xl border-white/10">
              <h3 className="text-[20px] font-semibold text-[#00dbe9] mb-4">System Types</h3>
              <div className="space-y-2 text-[12px]">
                <div className="text-[#b9cacb]">• Molecules</div>
                <div className="text-[#b9cacb]">• Quantum Circuits</div>
                <div className="text-[#b9cacb]">• Optimization Problems</div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
