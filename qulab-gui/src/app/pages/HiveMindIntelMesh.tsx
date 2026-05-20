import { useNavigate } from 'react-router';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function HiveMindIntelMesh() {
  const navigate = useNavigate();
  const { health, loading: healthLoading } = useLabHealth('global');

  // Generate node grid data
  const generateNodes = () => {
    const nodes = [];
    const colors = ['#00f0ff', '#8b5cf6', 'rgba(59,73,75,0.3)'];
    const weights = [0.6, 0.2, 0.2]; // 60% active, 20% syncing, 20% idle

    for (let i = 0; i < 150; i++) {
      const rand = Math.random();
      let color = colors[2]; // idle
      let glow = false;

      if (rand < weights[0]) {
        color = colors[0]; // active
        glow = true;
      } else if (rand < weights[0] + weights[1]) {
        color = colors[1]; // syncing
      }

      nodes.push({ id: i, color, glow });
    }

    return nodes;
  };

  const nodes = generateNodes();

  return (
    <>
      <style>{`
        body {
          min-height: max(884px, 100dvh);
          background-color: #050505;
          color: #e5e2e1;
          font-family: 'JetBrains Mono', monospace;
          margin: 0;
          overflow-x: hidden;
        }
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

      <div className="bg-[#0e0e0e] text-[#e5e2e1]">
        {/* TopAppBar */}
        <header className="fixed top-0 w-full z-50 flex justify-between items-center px-4 md:px-8 h-16 bg-[rgba(14,14,14,0.8)] backdrop-blur-xl border-b border-[rgba(59,73,75,0.3)]">
          <div className="flex items-center gap-4">
            <span className="material-symbols-outlined text-[#00dbe9]">grid_view</span>
            <h1 className="font-['Space_Grotesk'] text-[20px] leading-[1.2] tracking-tighter text-[#00dbe9]">HIVE MIND // INTEL MESH</h1>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 border border-[rgba(0,219,233,0.3)] text-[10px] font-bold text-[#00dbe9] tracking-widest bg-[rgba(0,219,233,0.05)]">NIST-V2 VERIFIED</span>
            <button className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#00dbe9] px-4 py-2 border border-[rgba(59,73,75,0.3)] hover:bg-[rgba(0,240,255,0.1)] transition-colors ring-1 ring-[#00dbe9]">SYSTEM LOCKED</button>
          </div>
        </header>

        {/* NavigationDrawer (Desktop Only) */}
        <aside className="hidden lg:flex flex-col fixed left-0 top-16 bottom-0 z-40 w-64 bg-[rgba(32,31,31,0.95)] backdrop-blur-md border-r border-[rgba(59,73,75,0.2)] p-4">
          <div className="mb-8 px-4">
            <p className="font-['JetBrains_Mono'] text-[10px] text-[rgba(185,202,203,0.5)] mb-2">SYSTEM_ROOT</p>
            <p className="font-['Space_Grotesk'] text-[20px] leading-[1.2] text-[#00dbe9]">TACTICAL_OS_V2</p>
          </div>
          <nav className="space-y-1">
            <a className="flex items-center gap-3 px-4 py-3 text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors cursor-pointer">
              <span className="material-symbols-outlined text-lg">visibility</span>
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">OVERWATCH</span>
            </a>
            <a className="flex items-center gap-3 px-4 py-3 text-[#00dbe9] bg-[rgba(0,240,255,0.1)] border-l-4 border-[#00dbe9] translate-x-1 transition-transform cursor-pointer">
              <span className="material-symbols-outlined text-lg">science</span>
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">LAB_STATUS</span>
            </a>
            <a className="flex items-center gap-3 px-4 py-3 text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors cursor-pointer">
              <span className="material-symbols-outlined text-lg">leak_add</span>
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">AGENT_TX</span>
            </a>
            <a className="flex items-center gap-3 px-4 py-3 text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors cursor-pointer">
              <span className="material-symbols-outlined text-lg">location_searching</span>
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">GRID_MAP</span>
            </a>
            <a className="flex items-center gap-3 px-4 py-3 text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors cursor-pointer">
              <span className="material-symbols-outlined text-lg">lock_open</span>
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">DECRYPTION</span>
            </a>
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
                  <span className="font-['JetBrains_Mono'] text-[10px] text-[#00dbe9] tracking-[0.2em]">NEURAL_THROUGHPUT</span>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#00dbe9] rounded-full animate-pulse"></span>
                    <span className="text-[10px] text-[#00dbe9] font-bold">NOMINAL</span>
                  </div>
                </div>
                <div className="mb-6">
                  <p className="font-['JetBrains_Mono'] text-5xl md:text-6xl text-[#00dbe9] font-bold tracking-tighter">142.8<span className="text-xl opacity-50">PF/S</span></p>
                  <p className="font-['JetBrains_Mono'] text-[12px] text-[#b9cacb] mt-2">HIVE COMPUTE CAPACITY</p>
                </div>
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-[10px] font-bold mb-2">
                    <span className="text-[#b9cacb]">RESILIENCE</span>
                    <span className="text-[#00dbe9]">98.4%</span>
                  </div>
                  <div className="segmented-progress">
                    {[...Array(10)].map((_, i) => (
                      <div key={i} className={`progress-block flex-1 ${i < 9 ? 'active' : ''}`}></div>
                    ))}
                  </div>
                </div>
              </div>
            </section>

            {/* Echo Reasoning Stream (Top Right) */}
            <section className="col-span-12 lg:col-span-8 glass-panel p-6 bg-[rgba(0,219,233,0.05)]">
              <div className="flex justify-between items-center mb-6 border-b border-[rgba(59,73,75,0.3)] pb-2">
                <span className="font-['JetBrains_Mono'] text-[10px] text-[#b9cacb] tracking-[0.2em]">ECHO_REASONING_STREAM</span>
                <span className="material-symbols-outlined text-[#00dbe9] text-sm">terminal</span>
              </div>
              <div className="font-['JetBrains_Mono'] text-sm leading-relaxed text-[#b9cacb] space-y-4 max-h-[160px] overflow-y-auto">
                <p className="border-l-2 border-[#00dbe9] pl-4 py-1">
                  <span className="text-[#00dbe9] font-bold">HIVE_INTEL:</span> SYNCHRONIZING PATHOGEN MODELS ACROSS SECTORS 4-9. LATTICE STABILITY AT OPTIMAL PARAMETERS.
                </p>
                <p className="border-l-2 border-[rgba(59,73,75,1)] pl-4 py-1">
                  <span className="text-[#b9cacb] font-bold opacity-50">SYSTEM_LOG [14:22:01]:</span> ALLOCATING SUB-PROCESSES TO LAB_211 FOR G-22 SEQUENCING.
                </p>
                <p className="border-l-2 border-[#00dbe9] pl-4 py-1">
                  <span className="text-[#00dbe9] font-bold">ECHO_V3:</span> DETECTED ANOMALY IN SECTOR 2. INITIATING REDUNDANT VALIDATION PROTOCOL.
                </p>
              </div>
            </section>

            {/* Global Intel Mesh (Central Visualization) */}
            <section className="col-span-12 lg:col-span-8 glass-panel p-6 relative min-h-[500px]">
              <div className="flex justify-between items-start mb-8">
                <div>
                  <span className="font-['JetBrains_Mono'] text-[10px] text-[#00dbe9] tracking-[0.2em]">GLOBAL_INTEL_MESH</span>
                  <h2 className="font-['Space_Grotesk'] text-[32px] leading-[1.2] text-[#e5e2e1]">220 LAB_CONCURRENT_LINK</h2>
                </div>
                <div className="flex gap-4">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#00f0ff] rounded-full"></span>
                    <span className="text-[10px] font-bold opacity-70">ACTIVE</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-[#8b5cf6] rounded-full"></span>
                    <span className="text-[10px] font-bold opacity-70">SYNCING</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-[rgba(59,73,75,1)] rounded-full"></span>
                    <span className="text-[10px] font-bold opacity-70">IDLE</span>
                  </div>
                </div>
              </div>
              <div className="node-grid">
                {nodes.map((node) => (
                  <div
                    key={node.id}
                    className={`node ${node.glow ? 'cyan-glow' : ''}`}
                    style={{ backgroundColor: node.color }}
                  ></div>
                ))}
              </div>
              <div className="absolute bottom-6 left-6 right-6 flex items-center justify-between border-t border-[rgba(59,73,75,0.3)] pt-4">
                <p className="text-[10px] text-[#b9cacb] font-bold tracking-widest uppercase">Target Vector: [42.08° N, 71.12° W]</p>
                <p className="text-[10px] text-[#00dbe9] font-bold tracking-widest">HIVE_STABILITY: OPTIMAL</p>
              </div>
            </section>

            {/* Live Agent Synchronization (Right Column) */}
            <section className="col-span-12 lg:col-span-4 glass-panel flex flex-col max-h-[500px]">
              <div className="p-6 border-b border-[rgba(59,73,75,0.3)]">
                <span className="font-['JetBrains_Mono'] text-[10px] text-[#00dbe9] tracking-[0.2em]">LIVE_AGENT_SYNC</span>
                <p className="font-['Space_Grotesk'] text-[20px] leading-[1.2] text-[#e5e2e1]">4,092 TOTAL AGENTS</p>
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-2">
                {/* Agent Items */}
                <div className="flex items-center justify-between p-3 border border-[rgba(59,73,75,0.2)] hover:border-[rgba(0,219,233,0.4)] transition-colors bg-white/5">
                  <div className="flex items-center gap-3">
                    <div className="w-1 h-8 bg-[#00f0ff]"></div>
                    <div>
                      <p className="text-[11px] font-bold text-[#e5e2e1]">AGENT_084</p>
                      <p className="text-[9px] text-[#b9cacb]">RE-SEQUENCING G-22</p>
                    </div>
                  </div>
                  <span className="text-[9px] text-[rgba(0,219,233,0.6)]">0.02ms</span>
                </div>
                <div className="flex items-center justify-between p-3 border border-[rgba(59,73,75,0.2)] hover:border-[rgba(0,219,233,0.4)] transition-colors bg-white/5">
                  <div className="flex items-center gap-3">
                    <div className="w-1 h-8 bg-[#8b5cf6]"></div>
                    <div>
                      <p className="text-[11px] font-bold text-[#e5e2e1]">AGENT_112</p>
                      <p className="text-[9px] text-[#b9cacb]">LATTICE STABILIZATION</p>
                    </div>
                  </div>
                  <span className="text-[9px] text-[rgba(0,219,233,0.6)]">0.14ms</span>
                </div>
                <div className="flex items-center justify-between p-3 border border-[rgba(59,73,75,0.2)] hover:border-[rgba(0,219,233,0.4)] transition-colors bg-white/5">
                  <div className="flex items-center gap-3">
                    <div className="w-1 h-8 bg-[#00f0ff]"></div>
                    <div>
                      <p className="text-[11px] font-bold text-[#e5e2e1]">AGENT_391</p>
                      <p className="text-[9px] text-[#b9cacb]">SYNAPTIC HANDSHAKE</p>
                    </div>
                  </div>
                  <span className="text-[9px] text-[rgba(0,219,233,0.6)]">0.05ms</span>
                </div>
                <div className="flex items-center justify-between p-3 border border-[rgba(59,73,75,0.2)] hover:border-[rgba(0,219,233,0.4)] transition-colors bg-white/5">
                  <div className="flex items-center gap-3">
                    <div className="w-1 h-8 bg-[rgba(59,73,75,1)]"></div>
                    <div>
                      <p className="text-[11px] font-bold text-[#e5e2e1]">AGENT_002</p>
                      <p className="text-[9px] text-[#b9cacb]">IDLE / BUFFERING</p>
                    </div>
                  </div>
                  <span className="text-[9px] text-[rgba(0,219,233,0.6)]">--</span>
                </div>
                <div className="flex items-center justify-between p-3 border border-[rgba(59,73,75,0.2)] hover:border-[rgba(0,219,233,0.4)] transition-colors bg-white/5">
                  <div className="flex items-center gap-3">
                    <div className="w-1 h-8 bg-[#00f0ff]"></div>
                    <div>
                      <p className="text-[11px] font-bold text-[#e5e2e1]">AGENT_923</p>
                      <p className="text-[9px] text-[#b9cacb]">PROTEIN FOLDING SIM</p>
                    </div>
                  </div>
                  <span className="text-[9px] text-[rgba(0,219,233,0.6)]">0.08ms</span>
                </div>
              </div>
              <div className="p-4 border-t border-[rgba(59,73,75,0.3)] text-center">
                <button className="text-[10px] font-bold text-[#00dbe9] hover:underline underline-offset-4 uppercase tracking-[0.2em]">VIEW ALL TRANSMISSIONS</button>
              </div>
            </section>
          </div>
        </main>

        {/* BottomNavBar */}
        <nav className="fixed bottom-0 w-full z-50 flex justify-around items-stretch h-16 bg-[rgba(14,14,14,0.9)] backdrop-blur-2xl border-t border-[rgba(59,73,75,0.3)] lg:hidden">
          <a className="flex flex-col items-center justify-center text-[rgba(185,202,203,0.6)] px-4 py-2 hover:text-[#00dbe9] hover:bg-white/5 transition-colors">
            <span className="material-symbols-outlined">map</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">FLEET</span>
          </a>
          <a className="flex flex-col items-center justify-center text-[#00dbe9] bg-[rgba(0,240,255,0.2)] border-t-2 border-[#00dbe9] px-4 py-2 scale-95 transition-transform duration-100">
            <span className="material-symbols-outlined">lan</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">MESH</span>
          </a>
          <a className="flex flex-col items-center justify-center text-[rgba(185,202,203,0.6)] px-4 py-2 hover:text-[#00dbe9] hover:bg-white/5 transition-colors">
            <span className="material-symbols-outlined">terminal</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">LOGS</span>
          </a>
          <a className="flex flex-col items-center justify-center text-[rgba(185,202,203,0.6)] px-4 py-2 hover:text-[#00dbe9] hover:bg-white/5 transition-colors">
            <span className="material-symbols-outlined">sync_alt</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">SYNC</span>
          </a>
        </nav>

        {/* Floating UI Decorations */}
        <div className="fixed top-24 right-8 pointer-events-none hidden xl:block opacity-20">
          <div className="text-[8px] font-bold space-y-1">
            <p>X-COORD: 42.1192</p>
            <p>Y-COORD: -71.2291</p>
            <p>Z-COORD: 1.0029</p>
            <div className="w-16 h-[0.5px] bg-[#00dbe9]"></div>
            <p>PACKET_LOSS: 0.00%</p>
          </div>
        </div>
      </div>
    </>
  );
}
