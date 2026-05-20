import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router';
import { apiClient } from '../../lib/api-client';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function HiveMindIntelMeshDashboard() {
  const navigate = useNavigate();
  const { health, loading: healthLoading } = useLabHealth('global');
  const [nodeStates, setNodeStates] = useState<string[]>([]);

  useEffect(() => {
    // Generate 220 lab nodes with random states
    const states = Array.from({ length: 220 }, () => {
      const rand = Math.random();
      if (rand < 0.6) return 'active';
      if (rand < 0.75) return 'syncing';
      return 'idle';
    });
    setNodeStates(states);
  }, []);

  const agents = [
    { id: 'AGENT_084', task: 'RE-SEQUENCING G-22', latency: '0.02ms', status: 'active' },
    { id: 'AGENT_112', task: 'LATTICE STABILIZATION', latency: '0.14ms', status: 'syncing' },
    { id: 'AGENT_391', task: 'SYNAPTIC HANDSHAKE', latency: '0.05ms', status: 'active' },
    { id: 'AGENT_002', task: 'IDLE / BUFFERING', latency: '--', status: 'idle' },
    { id: 'AGENT_923', task: 'PROTEIN FOLDING SIM', latency: '0.08ms', status: 'active' },
  ];

  const reasoningStream = [
    { type: 'HIVE_INTEL', text: 'SYNCHRONIZING PATHOGEN MODELS ACROSS SECTORS 4-9. LATTICE STABILITY AT OPTIMAL PARAMETERS.', highlight: true },
    { type: 'SYSTEM_LOG [14:22:01]', text: 'ALLOCATING SUB-PROCESSES TO LAB_211 FOR G-22 SEQUENCING.', highlight: false },
    { type: 'ECHO_V3', text: 'DETECTED ANOMALY IN SECTOR 2. INITIATING REDUNDANT VALIDATION PROTOCOL.', highlight: true },
  ];

  return (
    <>
      <style>{`
        .cyan-glow {
          box-shadow: 0 0 15px rgba(0, 219, 233, 0.3);
        }
        .segmented-progress {
          display: flex;
          gap: 2px;
        }
        .progress-block {
          width: 8px;
          height: 16px;
          background: rgba(0, 219, 233, 0.2);
        }
        .progress-block.active {
          background: #00dbe9;
          box-shadow: 0 0 8px #00dbe9;
        }
        .node-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(12px, 1fr));
          gap: 6px;
        }
        .node {
          width: 8px;
          height: 8px;
          border-radius: 1px;
        }
      `}</style>

      <div className="min-h-screen qulab-page-bg text-foreground font-['JetBrains_Mono']">
        {/* TopAppBar */}
        <header className="fixed top-0 w-full z-50 flex justify-between items-center px-4 md:px-8 h-16 bg-[#0e0e0e]/80 backdrop-blur-xl border-b border-[#3b494b]/30">
          <div className="flex items-center gap-4">
            <span className="material-symbols-outlined text-[#00dbe9]">grid_view</span>
            <h1 className="text-xl tracking-tighter text-[#00dbe9] font-semibold">HIVE MIND // INTEL MESH</h1>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 border border-[#00dbe9]/30 text-[10px] font-bold text-[#00dbe9] tracking-widest bg-[#00dbe9]/5">NIST-V2 VERIFIED</span>
            <button className="text-[11px] text-[#00dbe9] px-4 py-2 border border-[#3b494b]/30 hover:bg-[#00f0ff]/10 transition-colors ring-1 ring-[#00dbe9] font-bold tracking-[0.1em]">SYSTEM LOCKED</button>
          </div>
        </header>

        {/* NavigationDrawer (Desktop Only) */}
        <aside className="hidden lg:flex flex-col fixed left-0 top-16 bottom-0 z-40 w-64 bg-[#201f1f]/95 backdrop-blur-md border-r border-[#3b494b]/20 p-4">
          <div className="mb-8 px-4">
            <p className="text-[10px] text-[#b9cacb]/50 mb-2 font-bold tracking-[0.1em]">SYSTEM_ROOT</p>
            <p className="text-xl text-[#00dbe9] font-semibold">TACTICAL_OS_V2</p>
          </div>
          <nav className="space-y-1">
            <button onClick={() => navigate('/')} className="flex items-center gap-3 px-4 py-3 text-[#b9cacb]/70 hover:bg-[#00dbe9]/5 transition-colors w-full text-left">
              <span className="material-symbols-outlined text-lg">visibility</span>
              <span className="text-[11px] font-bold tracking-[0.1em]">OVERWATCH</span>
            </button>
            <button className="flex items-center gap-3 px-4 py-3 text-[#00dbe9] bg-[#00f0ff]/10 border-l-4 border-[#00dbe9] translate-x-1 transition-transform w-full text-left">
              <span className="material-symbols-outlined text-lg">science</span>
              <span className="text-[11px] font-bold tracking-[0.1em]">LAB_STATUS</span>
            </button>
            <button onClick={() => navigate('/agent-telemetry-deep-dive')} className="flex items-center gap-3 px-4 py-3 text-[#b9cacb]/70 hover:bg-[#00dbe9]/5 transition-colors w-full text-left">
              <span className="material-symbols-outlined text-lg">leak_add</span>
              <span className="text-[11px] font-bold tracking-[0.1em]">AGENT_TX</span>
            </button>
            <button className="flex items-center gap-3 px-4 py-3 text-[#b9cacb]/70 hover:bg-[#00dbe9]/5 transition-colors w-full text-left">
              <span className="material-symbols-outlined text-lg">location_searching</span>
              <span className="text-[11px] font-bold tracking-[0.1em]">GRID_MAP</span>
            </button>
            <button onClick={() => navigate('/system-lockdown')} className="flex items-center gap-3 px-4 py-3 text-[#b9cacb]/70 hover:bg-[#00dbe9]/5 transition-colors w-full text-left">
              <span className="material-symbols-outlined text-lg">lock_open</span>
              <span className="text-[11px] font-bold tracking-[0.1em]">DECRYPTION</span>
            </button>
          </nav>
        </aside>

        {/* Main Content Canvas */}
              <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-20 pb-24 lg:pl-64 px-4 md:px-8 max-w-[1600px] mx-auto min-h-screen">
          <div className="grid grid-cols-12 gap-4">
            {/* Neural Throughput Section (Top Left) */}
            <section className="col-span-12 lg:col-span-4 glass-panel p-6 flex flex-col justify-between">
              <div>
                <div className="flex justify-between items-start mb-4">
                  <span className="text-[10px] text-[#00dbe9] tracking-[0.2em] font-bold">NEURAL_THROUGHPUT</span>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#00dbe9] rounded-full animate-pulse"></span>
                    <span className="text-[10px] text-[#00dbe9] font-bold">NOMINAL</span>
                  </div>
                </div>
                <div className="mb-6">
                  <p className="text-5xl md:text-6xl text-[#00dbe9] font-bold tracking-tighter">142.8<span className="text-xl opacity-50">PF/S</span></p>
                  <p className="text-[12px] text-[#b9cacb] mt-2 font-bold tracking-[0.1em]">HIVE COMPUTE CAPACITY</p>
                </div>
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-[10px] font-bold mb-2">
                    <span className="text-[#b9cacb] tracking-[0.1em]">RESILIENCE</span>
                    <span className="text-[#00dbe9] tracking-[0.1em]">98.4%</span>
                  </div>
                  <div className="segmented-progress">
                    {Array.from({ length: 10 }).map((_, i) => (
                      <div key={i} className={`progress-block flex-1 ${i < 9 ? 'active' : ''}`}></div>
                    ))}
                  </div>
                </div>
              </div>
            </section>

            {/* Echo Reasoning Stream (Top Right) */}
            <section className="col-span-12 lg:col-span-8 glass-panel p-6 bg-[#00dbe9]/5">
              <div className="flex justify-between items-center mb-6 border-b border-[#3b494b]/30 pb-2">
                <span className="text-[10px] text-[#b9cacb] tracking-[0.2em] font-bold">ECHO_REASONING_STREAM</span>
                <span className="material-symbols-outlined text-[#00dbe9] text-sm">terminal</span>
              </div>
              <div className="text-sm leading-relaxed text-[#b9cacb] space-y-4 max-h-[160px] overflow-y-auto">
                {reasoningStream.map((log, i) => (
                  <p key={i} className={`border-l-2 pl-4 py-1 ${log.highlight ? 'border-[#00dbe9]' : 'border-[#3b494b]'}`}>
                    <span className={`font-bold ${log.highlight ? 'text-[#00dbe9]' : 'text-[#b9cacb] opacity-50'}`}>{log.type}:</span> {log.text}
                  </p>
                ))}
              </div>
            </section>

            {/* Global Intel Mesh (Central Visualization) */}
            <section className="col-span-12 lg:col-span-8 glass-panel p-6 relative min-h-[500px]">
              <div className="flex justify-between items-start mb-8">
                <div>
                  <span className="text-[10px] text-[#00dbe9] tracking-[0.2em] font-bold">GLOBAL_INTEL_MESH</span>
                  <h2 className="text-3xl text-[#e5e2e1] font-semibold">220 LAB_CONCURRENT_LINK</h2>
                </div>
                <div className="flex gap-4">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#00f0ff] rounded-full"></span>
                    <span className="text-[10px] font-bold opacity-70 tracking-[0.1em]">ACTIVE</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#8b5cf6] rounded-full"></span>
                    <span className="text-[10px] font-bold opacity-70 tracking-[0.1em]">SYNCING</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#3b494b] rounded-full"></span>
                    <span className="text-[10px] font-bold opacity-70 tracking-[0.1em]">IDLE</span>
                  </div>
                </div>
              </div>
              <div className="node-grid">
                {nodeStates.map((state, i) => (
                  <div
                    key={i}
                    className={`node ${
                      state === 'active' ? 'bg-[#00f0ff] cyan-glow' :
                      state === 'syncing' ? 'bg-[#8b5cf6]' :
                      'bg-[#3b494b]/30'
                    }`}
                  ></div>
                ))}
              </div>
              <div className="absolute bottom-6 left-6 right-6 flex items-center justify-between border-t border-[#3b494b]/30 pt-4">
                <p className="text-[10px] text-[#b9cacb] font-bold tracking-widest uppercase">Target Vector: [42.08° N, 71.12° W]</p>
                <p className="text-[10px] text-[#00dbe9] font-bold tracking-widest">HIVE_STABILITY: OPTIMAL</p>
              </div>
            </section>

            {/* Live Agent Synchronization (Right Column) */}
            <section className="col-span-12 lg:col-span-4 glass-panel flex flex-col max-h-[500px]">
              <div className="p-6 border-b border-[#3b494b]/30">
                <span className="text-[10px] text-[#00dbe9] tracking-[0.2em] font-bold">LIVE_AGENT_SYNC</span>
                <p className="text-xl text-[#e5e2e1] font-semibold">4,092 TOTAL AGENTS</p>
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-2">
                {agents.map((agent) => (
                  <div key={agent.id} className="flex items-center justify-between p-3 border border-[#3b494b]/20 hover:border-[#00dbe9]/40 transition-colors bg-white/5">
                    <div className="flex items-center gap-3">
                      <div className={`w-1 h-8 ${
                        agent.status === 'active' ? 'bg-[#00f0ff]' :
                        agent.status === 'syncing' ? 'bg-[#8b5cf6]' :
                        'bg-[#3b494b]'
                      }`}></div>
                      <div>
                        <p className="text-[11px] font-bold text-[#e5e2e1]">{agent.id}</p>
                        <p className="text-[9px] text-[#b9cacb]">{agent.task}</p>
                      </div>
                    </div>
                    <span className="text-[9px] text-[#00dbe9]/60">{agent.latency}</span>
                  </div>
                ))}
              </div>
              <div className="p-4 border-t border-[#3b494b]/30 text-center">
                <button className="text-[10px] font-bold text-[#00dbe9] hover:underline underline-offset-4 uppercase tracking-[0.2em]">VIEW ALL TRANSMISSIONS</button>
              </div>
            </section>
          </div>
        </main>

        {/* BottomNavBar */}
        <nav className="fixed bottom-0 w-full z-50 flex justify-around items-stretch h-16 bg-[#0e0e0e]/90 backdrop-blur-2xl border-t border-[#3b494b]/30 lg:hidden">
          <button className="flex flex-col items-center justify-center text-[#b9cacb]/60 px-4 py-2 hover:text-[#00dbe9] hover:bg-white/5 transition-colors">
            <span className="material-symbols-outlined">map</span>
            <span className="text-[11px] font-bold tracking-[0.1em]">FLEET</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#00dbe9] bg-[#00f0ff]/20 border-t-2 border-[#00dbe9] px-4 py-2 scale-95 transition-transform duration-100">
            <span className="material-symbols-outlined">lan</span>
            <span className="text-[11px] font-bold tracking-[0.1em]">MESH</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#b9cacb]/60 px-4 py-2 hover:text-[#00dbe9] hover:bg-white/5 transition-colors">
            <span className="material-symbols-outlined">terminal</span>
            <span className="text-[11px] font-bold tracking-[0.1em]">LOGS</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#b9cacb]/60 px-4 py-2 hover:text-[#00dbe9] hover:bg-white/5 transition-colors">
            <span className="material-symbols-outlined">sync_alt</span>
            <span className="text-[11px] font-bold tracking-[0.1em]">SYNC</span>
          </button>
        </nav>

        {/* Floating UI Decorations */}
        <div className="fixed top-24 right-8 pointer-events-none hidden xl:block opacity-20">
          <div className="text-[8px] font-bold space-y-1">
            <p>X-COORD: 42.1192</p>
            <p>Y-COORD: -71.2291</p>
            <p>Z-COORD: 1.0029</p>
            <div className="w-16 h-[0.5px] bg-[#00dbe9]"></div>
          </div>
        </div>
      </div>
    </>
  );
}
