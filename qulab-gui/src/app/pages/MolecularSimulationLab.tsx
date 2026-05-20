import { useNavigate } from 'react-router';
import { Image3DViewer } from '../components/Image3DViewer';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function MolecularSimulationLab() {
  const navigate = useNavigate();
  const { health, loading: healthLoading } = useLabHealth('chemistry');

  return (
    <>
      <style>{`
        .cyan-glow {
          box-shadow: 0 0 15px rgba(0, 219, 233, 0.15);
        }
        .scan-line {
          background: linear-gradient(to bottom, transparent 50%, rgba(0, 219, 233, 0.05) 50%);
          background-size: 100% 4px;
        }
        .segmented-progress {
          display: flex;
          gap: 2px;
        }
        .segment {
          height: 8px;
          flex: 1;
          background: rgba(255, 255, 255, 0.1);
        }
        .segment.active {
          background: #00dbe9;
        }
      `}</style>

      <div className="bg-[#131313] text-[#e5e2e1] font-['JetBrains_Mono'] selection:bg-[rgba(219,252,255,0.3)] min-h-screen overflow-x-hidden">
        {/* TopAppBar Shell */}
        <header className="fixed top-0 w-full z-50 bg-[rgba(19,19,19,0.8)] backdrop-blur-xl border-b border-[rgba(59,73,75,0.3)] flex justify-between items-center px-4 md:px-8 py-2">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-[#00dbe9]">terminal</span>
            <span className="font-['Space_Grotesk'] text-[20px] leading-[1.2] font-semibold tracking-tighter text-[#00dbe9] uppercase">QULAB_INFINITE_OS</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="hidden md:flex gap-6 items-center">
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] font-bold text-[#dbfcff] cursor-pointer hover:bg-[rgba(219,252,255,0.1)] px-2 py-1 transition-colors">TERMINAL_ACTIVE</span>
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] font-bold text-[#849495] cursor-pointer hover:bg-[rgba(219,252,255,0.1)] px-2 py-1 transition-colors">SECURE_CHANNEL</span>
            </div>
            <span className="material-symbols-outlined text-[#00dbe9]">encrypted</span>
          </div>
        </header>

              <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-20 pb-24 px-4 md:px-8 max-w-[1600px] mx-auto grid grid-cols-12 gap-2">
          {/* Dashboard Navigation Context (Left Sidebar Simulation) */}
          <div className="hidden lg:flex col-span-1 flex-col gap-2 sticky top-24 h-[calc(100vh-160px)]">
            <div className="flex flex-col items-center justify-center p-2 border border-[rgba(219,252,255,0.2)] bg-[rgba(219,252,255,0.05)] text-[#dbfcff]">
              <span className="material-symbols-outlined">biotech</span>
              <span className="text-[9px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold mt-1">LABS</span>
            </div>
            <div onClick={() => navigate('/')} className="flex flex-col items-center justify-center p-2 text-[#849495] hover:text-[#dbfcff] transition-all cursor-pointer">
              <span className="material-symbols-outlined">grid_view</span>
              <span className="text-[9px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold mt-1">DASH</span>
            </div>
            <div onClick={() => navigate('/mission')} className="flex flex-col items-center justify-center p-2 text-[#849495] hover:text-[#dbfcff] transition-all cursor-pointer">
              <span className="material-symbols-outlined">target</span>
              <span className="text-[9px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold mt-1">MISSION</span>
            </div>
            <div onClick={() => navigate('/system')} className="mt-auto flex flex-col items-center justify-center p-2 text-[#849495] hover:text-[#dbfcff] transition-all cursor-pointer">
              <span className="material-symbols-outlined">settings</span>
              <span className="text-[9px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold mt-1">SYSTEM</span>
            </div>
          </div>

          {/* Main Telemetry Grid */}
          <div className="col-span-12 lg:col-span-8 flex flex-col gap-2">
            {/* Central Visualizer Canvas */}
            <section className="glass-panel relative aspect-video w-full overflow-hidden border border-[rgba(59,73,75,0.3)] flex items-center justify-center scan-line">
              <div className="absolute top-4 left-4 z-10 flex flex-col">
                <span className="text-[#dbfcff] font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] font-bold">VISUALIZER_STREAM_01</span>
                <span className="text-[#849495] text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold">RENDER: REAL-TIME 3D_MOLECULAR</span>
              </div>
              <div className="absolute top-4 right-4 z-10">
                <div className="flex items-center gap-2 px-2 py-1 bg-[rgba(219,252,255,0.2)] border border-[#dbfcff] text-[#dbfcff] font-['JetBrains_Mono'] text-[10px]">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#dbfcff] animate-pulse"></span>
                  LIVE_SIMULATION
                </div>
              </div>
              {/* Simulation Visual */}
              <Image3DViewer
                imageUrl="https://lh3.googleusercontent.com/aida/ADBb0uig-sDxI2nM2Er9xCQ92drjKIOXkce1AATGbyvZdym3RRHR3XxDS9kMH1IEIswKDv13Wxzs6I90pwlk6wr-7g31fnijQCSGsNMl0lq49D1cIQ_yJsDOgYb6xusUyFb9g_s5D1_CpNjXs7TDFz9bIZyfPV8trd4YwwUSkJ68_6-utPcenhXktkq8J34aT8cBGiOE68w65Xa7nft7ujTyJAeco80GHeCGc-pDK4HNSgAIW5-s7Mtnp0x4LVs"
                alt="Tactical 3D molecular simulation visualization"
                className="w-full h-full object-cover opacity-80 mix-blend-screen"
                autoRotate={true}
              />
              {/* Overlay Telemetry Lines */}
              <div className="absolute inset-0 pointer-events-none border-[1px] border-[rgba(219,252,255,0.05)]">
                <div className="absolute top-1/2 left-0 w-full h-[1px] bg-[rgba(219,252,255,0.1)]"></div>
                <div className="absolute top-0 left-1/2 w-[1px] h-full bg-[rgba(219,252,255,0.1)]"></div>
              </div>
            </section>

            {/* Bottom Data Cluster */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {/* Port Block */}
              <div className="glass-panel p-3 border border-[rgba(59,73,75,0.3)]">
                <div className="flex justify-between items-start mb-2">
                  <span className="text-[#849495] font-['JetBrains_Mono'] text-[10px]">PORT_NODE</span>
                  <span className="w-2 h-2 bg-[#00dbe9]"></span>
                </div>
                <div className="font-['JetBrains_Mono'] text-[24px] leading-[1] tracking-[-0.05em] font-medium text-[#dbfcff]">8080</div>
                <div className="text-[9px] font-['JetBrains_Mono'] text-[#3b494b] mt-1">TCP/UDP_ENCRYPTED</div>
              </div>

              {/* Subject Block */}
              <div className="glass-panel p-3 border border-[rgba(59,73,75,0.3)]">
                <div className="flex justify-between items-start mb-2">
                  <span className="text-[#849495] font-['JetBrains_Mono'] text-[10px]">SUBJECT_ID</span>
                  <span className="w-2 h-2 bg-[#72ff70]"></span>
                </div>
                <div className="font-['JetBrains_Mono'] text-lg text-[#dbfcff] truncate">ORGANIC_SYN</div>
                <div className="text-[9px] font-['JetBrains_Mono'] text-[#3b494b] mt-1">CAT: BIO_CHEMICAL</div>
              </div>

              {/* Uptime Block */}
              <div className="glass-panel p-3 border border-[rgba(59,73,75,0.3)]">
                <div className="flex justify-between items-start mb-2">
                  <span className="text-[#849495] font-['JetBrains_Mono'] text-[10px]">UPTIME_SYNC</span>
                  <span className="w-2 h-2 bg-[#dbfcff]"></span>
                </div>
                <div className="font-['JetBrains_Mono'] text-[24px] leading-[1] tracking-[-0.05em] font-medium text-[#dbfcff]">142:08:44</div>
                <div className="text-[9px] font-['JetBrains_Mono'] text-[#3b494b] mt-1">STABILITY: 99.98%</div>
              </div>

              {/* Reagent Block */}
              <div className="glass-panel p-3 border border-[rgba(59,73,75,0.3)]">
                <div className="flex justify-between items-start mb-2">
                  <span className="text-[#849495] font-['JetBrains_Mono'] text-[10px]">REAGENT_LVL</span>
                  <span className="w-2 h-2 bg-[#ffb4ab]"></span>
                </div>
                <div className="font-['JetBrains_Mono'] text-[24px] leading-[1] tracking-[-0.05em] font-medium text-[#ffb4ab]">12.4%</div>
                <div className="segmented-progress mt-2">
                  <div className="segment active"></div>
                  <div className="segment"></div>
                  <div className="segment"></div>
                  <div className="segment"></div>
                  <div className="segment"></div>
                  <div className="segment"></div>
                  <div className="segment"></div>
                  <div className="segment"></div>
                </div>
              </div>
            </div>
          </div>

          {/* Echo AGI Reasoning Log (Right Panel) */}
          <aside className="col-span-12 lg:col-span-3 flex flex-col gap-2">
            <div className="glass-panel flex-1 flex flex-col border border-[rgba(59,73,75,0.3)] h-[calc(100vh-160px)] md:h-auto lg:h-[calc(100vh-160px)] overflow-hidden">
              <div className="p-4 border-b border-[rgba(59,73,75,0.3)] flex justify-between items-center bg-[#1c1b1b]">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-[#dbfcff] scale-75">neurology</span>
                  <h3 className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] font-bold text-[#dbfcff]">ECHO_AGI_LOG</h3>
                </div>
                <span className="text-[9px] font-['JetBrains_Mono'] px-1 bg-[#72ff70] text-[#003907]">ACTIVE</span>
              </div>
              <div className="p-4 flex-1 overflow-y-auto space-y-4 font-['JetBrains_Mono'] text-[12px] leading-relaxed text-[#b9cacb]">
                <div className="border-l-2 border-[rgba(219,252,255,0.4)] pl-3">
                  <span className="block text-[#dbfcff] font-bold mb-1">[08:44:12] THOUGHT_PROCESS:</span>
                  Analyzing molecular bonds in current simulation. Potential instability detected at Carbon-14 node. Initiating thermal compensation protocols.
                </div>
                <div className="border-l-2 border-[rgba(59,73,75,0.4)] pl-3">
                  <span className="block text-[#849495] font-bold mb-1">[08:44:15] MONITORING:</span>
                  Calibrating Reagent Flow-Rate. Pressure at 4.2 Bar. Within safety parameters for NIST_800_171.
                </div>
                <div className="border-l-2 border-[rgba(219,252,255,0.4)] pl-3">
                  <span className="block text-[#dbfcff] font-bold mb-1">[08:44:18] INFERENCE:</span>
                  Probability of yield success: 94.2%. Recommending increase in catalytic agent by 0.5mg to stabilize the secondary bond.
                </div>
                <div className="border-l-2 border-[rgba(114,255,112,0.4)] pl-3">
                  <span className="block text-[#72ff70] font-bold mb-1">[08:44:22] VERIFICATION:</span>
                  NIST-verified compliance badge issued for session 8080-ORG. Integrity check: PASSED.
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-[#dbfcff]">&gt;_</span>
                  <span className="w-1.5 h-4 bg-[#dbfcff] animate-pulse"></span>
                </div>
              </div>
              {/* Compliance Badge Footer */}
              <div className="p-3 border-t border-[rgba(59,73,75,0.3)] bg-[#0e0e0e]">
                <div className="flex items-center gap-2 mb-2">
                  <span className="material-symbols-outlined text-[14px] text-[#72ff70]">verified</span>
                  <span className="font-['JetBrains_Mono'] text-[9px] text-[#b9cacb]">NIST-VERIFIED COMPLIANCE</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <button className="bg-[#201f1f] border border-[rgba(59,73,75,0.3)] py-2 px-3 text-[10px] font-['JetBrains_Mono'] text-[#b9cacb] hover:border-[#dbfcff] transition-colors flex items-center justify-center gap-2">
                    <span className="material-symbols-outlined text-[14px]">download</span>
                    EX_REPORT
                  </button>
                  <button className="bg-[rgba(219,252,255,0.1)] border border-[rgba(219,252,255,0.4)] py-2 px-3 text-[10px] font-['JetBrains_Mono'] text-[#dbfcff] hover:bg-[rgba(219,252,255,0.2)] transition-all flex items-center justify-center gap-2 cyan-glow">
                    <span className="material-symbols-outlined text-[14px]">bolt</span>
                    OVERRIDE
                  </button>
                </div>
              </div>
            </div>
          </aside>
        </main>

        {/* Bottom Navigation Shell */}
        <nav className="fixed bottom-0 w-full z-50 bg-[rgba(32,31,31,0.6)] backdrop-blur-2xl border-t border-[rgba(59,73,75,0.3)] flex justify-around items-center h-16">
          <div onClick={() => navigate('/')} className="flex flex-col items-center justify-center text-[#849495] hover:text-[#7df4ff] transition-all cursor-pointer">
            <span className="material-symbols-outlined">grid_view</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] font-bold">DASHBOARD</span>
          </div>
          <div className="flex flex-col items-center justify-center text-[#00dbe9] bg-[rgba(219,252,255,0.05)] rounded-none border-t-2 border-[#dbfcff] h-full px-4">
            <span className="material-symbols-outlined">biotech</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] font-bold">LABS</span>
          </div>
          <div onClick={() => navigate('/mission')} className="flex flex-col items-center justify-center text-[#849495] hover:text-[#7df4ff] transition-all cursor-pointer">
            <span className="material-symbols-outlined">target</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] font-bold">MISSION</span>
          </div>
          <div onClick={() => navigate('/system')} className="flex flex-col items-center justify-center text-[#849495] hover:text-[#7df4ff] transition-all cursor-pointer">
            <span className="material-symbols-outlined">settings</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] font-bold">SYSTEM</span>
          </div>
        </nav>
      </div>
    </>
  );
}
