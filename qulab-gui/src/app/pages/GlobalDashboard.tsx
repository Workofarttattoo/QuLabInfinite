import { useNavigate } from 'react-router';
import { Image3DViewer } from '../components/Image3DViewer';
import { useLabsHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function GlobalDashboard() {
  const navigate = useNavigate();
  const { labsStatus, loading } = useLabsHealth();

  // Calculate system-wide metrics
  const totalLabs = Object.keys(labsStatus).length;
  const healthyLabs = Object.values(labsStatus).filter((lab) => lab.healthy).length;
  const systemStability = totalLabs > 0 ? ((healthyLabs / totalLabs) * 100).toFixed(2) : '0.00';

  return (
    <div className="bg-[#131313] text-[#e5e2e1] font-['JetBrains_Mono'] overflow-hidden selection:bg-[#00dbe9] selection:text-[#00363a]">
      <style>
        {`
          .scanline {
            background: linear-gradient(to bottom, transparent 50%, rgba(0, 219, 233, 0.03) 50%);
            background-size: 100% 4px;
          }
          ::-webkit-scrollbar { width: 4px; }
          ::-webkit-scrollbar-track { background: #050505; }
          ::-webkit-scrollbar-thumb { background: #3b494b; }
        `}
      </style>

      <div className="fixed inset-0 scanline pointer-events-none opacity-20 z-50"></div>

      {/* TopAppBar */}
      <header className="flex justify-between items-center w-full px-4 md:px-8 py-2 border-b border-[#3b494b]/50 bg-[#131313]/80 backdrop-blur-xl fixed top-0 z-40">
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-[#00dbe9]">terminal</span>
          <h1 className="text-[20px] font-bold text-[#00dbe9] tracking-tighter font-['Space_Grotesk']">QULAB_INF_OS // V.1.0.4</h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-4 text-[11px] font-bold tracking-[0.1em]">
            <span className="text-[#00dbe9]">SYSTEM_STABLE</span>
            <span className="text-[#b9cacb]/40">|</span>
            <span className="text-[#b9cacb]">LATENCY: 14MS</span>
          </div>
          <button className="px-3 py-1 border border-[#00dbe9]/50 text-[#00dbe9] text-[11px] font-bold tracking-[0.1em] hover:bg-[#00dbe9]/10 transition-colors">
            [ENCRYPTED]
          </button>
        </div>
      </header>

      {/* NavigationDrawer (Desktop Only) */}
      <aside className="hidden md:flex flex-col p-4 space-y-2 h-full w-80 fixed left-0 top-0 pt-20 bg-[#2a2a2a]/80 backdrop-blur-2xl border-r border-[#3b494b]/20 z-30">
        <div className="px-2 mb-4">
          <h2 className="text-[11px] font-bold tracking-[0.1em] text-[#00e639] tracking-widest">ECHO_INTEL_OVERLAY</h2>
        </div>
        <nav className="flex flex-col space-y-1">
          <a className="flex items-center gap-4 px-4 py-3 bg-[#13ff43]/10 text-[#72ff70] font-bold border-l-4 border-[#72ff70] hover:bg-[#e5e2e1]/5 transition-all cursor-pointer">
            <span className="material-symbols-outlined">settings_ethernet</span>
            <span className="text-[14px] leading-relaxed">STREAM_01</span>
          </a>
          <a className="flex items-center gap-4 px-4 py-3 text-[#e5e2e1]/50 text-[14px] leading-relaxed hover:bg-[#e5e2e1]/5 transition-all cursor-pointer">
            <span className="material-symbols-outlined">analytics</span>
            <span>TELEMETRY</span>
          </a>
          <a className="flex items-center gap-4 px-4 py-3 text-[#e5e2e1]/50 text-[14px] leading-relaxed hover:bg-[#e5e2e1]/5 transition-all cursor-pointer">
            <span className="material-symbols-outlined">history_edu</span>
            <span>COMMS_LOG</span>
          </a>
          <a className="flex items-center gap-4 px-4 py-3 text-[#e5e2e1]/50 text-[14px] leading-relaxed hover:bg-[#e5e2e1]/5 transition-all cursor-pointer">
            <span className="material-symbols-outlined">psychology</span>
            <span>NEURAL_MAP</span>
          </a>
        </nav>
        <div className="mt-auto border-t border-[#3b494b]/20 pt-4 px-2">
          <div className="glass-panel p-3 border border-[#3b494b]/30">
            <div className="flex justify-between items-start mb-2">
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#b9cacb]">ACTIVE_SESSION</span>
              <span className="w-2 h-2 bg-[#00e639] rounded-full animate-pulse"></span>
            </div>
            <p className="text-[10px] text-[#b9cacb] leading-tight opacity-70">
              Neural bridge established. Uplink verified via RSA-4096 cluster.
            </p>
          </div>
        </div>
      </aside>

      {/* Main Content Canvas */}
            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-20 pb-20 md:pb-8 md:pl-80 min-h-screen">
        <div className="p-4 grid grid-cols-12 gap-2">
          {/* Global Operational Status / World Map */}
          <section className="col-span-12 lg:col-span-8 glass-panel relative overflow-hidden group">
            <div className="p-4 border-b border-[#3b494b]/30 flex justify-between items-center">
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-bold tracking-[0.1em] text-[#00dbe9]">STATUS_MAP::GLOBAL_NODES</span>
              </div>
              <div className="flex gap-4">
                <span className="text-[11px] font-bold tracking-[0.1em] text-[#00e639] flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-[#00e639]"></span> ONLINE
                </span>
                <span className="text-[11px] font-bold tracking-[0.1em] text-[#b9cacb] flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-[#b9cacb]"></span> SYNCING
                </span>
              </div>
            </div>
            <div className="relative h-[340px] md:h-[450px] bg-[#0e0e0e]/50">
              {/* 3D World Map Viewer */}
              <Image3DViewer
                imageUrl="https://lh3.googleusercontent.com/aida-public/AB6AXuACz78O4FoIaEtcOJDp_BhL0m1x9Zi4xxBYQS-aBzq3jCz_v-oyBWRuQ43fWzS0ZPAzZJjX3FveWK_j_MoKtyoXFfuxTqIlxh6O7sUvJI1PpUX4wBbCfjRRb4LNsldB2ABFSzr2up32bkxLoVv7sKG3UW7bNdtsSV2fu2CbkwaSkCiYrdnQG-50HPxB3Ylfue3L09TerjFjBZHoEMumBYbGdbe2tKx-sl_L_kXn7xdBvbX8Lqh4fBb319EWdi_VQXz0zRJdefEB3Ds"
                alt="Global world map with data points"
                className="w-full h-full opacity-40"
                autoRotate={true}
              />

              {/* Node Callouts */}
              <div className="absolute top-[25%] left-[20%] group/node">
                <div className="w-3 h-3 bg-[#00dbe9] rounded-full border-glow-cyan"></div>
                <div className="absolute top-4 left-4 glass-panel px-2 py-1 border border-[#00dbe9]/30">
                  <span className="text-[11px] font-bold tracking-[0.1em] text-[#00dbe9]">NA_HUB_ALPHA</span>
                </div>
              </div>
              <div className="absolute top-[30%] left-[48%] group/node">
                <div className="w-3 h-3 bg-[#00dbe9] rounded-full border-glow-cyan"></div>
                <div className="absolute top-4 left-4 glass-panel px-2 py-1 border border-[#00dbe9]/30">
                  <span className="text-[11px] font-bold tracking-[0.1em] text-[#00dbe9]">EU_CENTRAL_ARRAY</span>
                </div>
              </div>
              <div className="absolute top-[55%] left-[80%] group/node">
                <div className="w-3 h-3 bg-[#00dbe9] rounded-full border-glow-cyan"></div>
                <div className="absolute top-4 left-4 glass-panel px-2 py-1 border border-[#00dbe9]/30">
                  <span className="text-[11px] font-bold tracking-[0.1em] text-[#00dbe9]">APAC_OFFSHORE_NODE</span>
                </div>
              </div>

              {/* Overlay Data Streams */}
              <div className="absolute bottom-4 right-4 text-right">
                <div className="text-[24px] leading-none tracking-[-0.05em] font-medium text-[#00dbe9] mb-1">STABILITY: {systemStability}%</div>
                <div className="text-[11px] font-bold tracking-[0.1em] text-[#b9cacb]">ACTIVE_TRANSFERS: 4,029/SEC</div>
              </div>
            </div>
          </section>

          {/* SAAG Tiles Stack */}
          <div className="col-span-12 lg:col-span-4 flex flex-col gap-2">
            {/* Fleet Utilization */}
            <article className="glass-panel p-4 flex flex-col justify-between border-glow-cyan bg-[#00f0ff]/5">
              <div className="flex justify-between items-start">
                <span className="text-[11px] font-bold tracking-[0.1em] text-[#b9cacb]">FLEET_UTILIZATION</span>
                <span className="material-symbols-outlined text-[#00dbe9]">rocket_launch</span>
              </div>
              <div className="mt-4">
                <div className="text-[24px] leading-none tracking-[-0.05em] font-medium text-[#00dbe9]">84.2%</div>
                <div className="w-full h-2 bg-[#3b494b]/30 mt-2 flex gap-1">
                  <div className="h-full bg-[#00dbe9] w-[10%]"></div>
                  <div className="h-full bg-[#00dbe9] w-[10%]"></div>
                  <div className="h-full bg-[#00dbe9] w-[10%]"></div>
                  <div className="h-full bg-[#00dbe9] w-[10%]"></div>
                  <div className="h-full bg-[#00dbe9] w-[10%]"></div>
                  <div className="h-full bg-[#00dbe9] w-[10%]"></div>
                  <div className="h-full bg-[#00dbe9] w-[10%]"></div>
                  <div className="h-full bg-[#00dbe9] w-[10%]"></div>
                  <div className="h-full bg-[#3b494b]/30 w-[10%]"></div>
                  <div className="h-full bg-[#3b494b]/30 w-[10%]"></div>
                </div>
              </div>
            </article>

            {/* Qubit Stability */}
            <article className="glass-panel p-4 flex flex-col justify-between border border-[#72ff70]/30 bg-[#13ff43]/5">
              <div className="flex justify-between items-start">
                <span className="text-[11px] font-bold tracking-[0.1em] text-[#b9cacb]">QUBIT_STABILITY</span>
                <span className="material-symbols-outlined text-[#00e639]">bolt</span>
              </div>
              <div className="mt-4">
                <div className="text-[24px] leading-none tracking-[-0.05em] font-medium text-[#00e639]">OPTIMAL</div>
                <div className="text-[11px] font-bold tracking-[0.1em] text-[#00e639]/60 mt-1">COHERENCE: 412MS</div>
              </div>
            </article>

            {/* Global Nodes */}
            <article className="glass-panel p-4 flex flex-col justify-between">
              <div className="flex justify-between items-start">
                <span className="text-[11px] font-bold tracking-[0.1em] text-[#b9cacb]">GLOBAL_NODES</span>
                <span className="material-symbols-outlined text-[#b9cacb]">public</span>
              </div>
              <div className="mt-4 flex items-end justify-between">
                <div>
                  <div className="text-[24px] leading-none tracking-[-0.05em] font-medium text-[#e5e2e1]">{totalLabs}</div>
                  <div className="text-[11px] font-bold tracking-[0.1em] text-[#b9cacb] mt-1">TOTAL_LABS</div>
                </div>
                <div className="text-right">
                  <div className="text-[24px] leading-none tracking-[-0.05em] font-medium text-[#00e639]">{healthyLabs}</div>
                  <div className="text-[11px] font-bold tracking-[0.1em] text-[#00e639] mt-1">HEALTHY</div>
                </div>
              </div>
            </article>
          </div>

          {/* Echo AGI Command Interface */}
          <section className="col-span-12 md:col-span-7 glass-panel flex flex-col">
            <div className="p-4 border-b border-[#3b494b]/30 flex items-center gap-3">
              <div className="w-2 h-6 bg-[#00dbe9]"></div>
              <h3 className="text-[20px] font-bold font-['Space_Grotesk']">ECHO_AGI::COMMAND_DIRECTIVES</h3>
            </div>
            <div className="p-4 space-y-4">
              <div className="bg-[#0e0e0e] p-4 border-l-4 border-[#00dbe9]">
                <div className="text-[11px] font-bold tracking-[0.1em] text-[#00dbe9] mb-2">ACTIVE_MISSION: NEURAL_RECLAMATION</div>
                <p className="text-[14px] leading-relaxed text-[#e5e2e1]/80">
                  Coordinate decentralized synthesis across all APAC offshore clusters. Prioritize data integrity over latency for batch #409-Z. Grounded intelligence suggests local atmospheric interference at North America Hub Alpha; rerouting telemetry through EU Central Array.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 border border-[#3b494b]/30">
                  <span className="text-[11px] font-bold tracking-[0.1em] text-[#b9cacb] block mb-1">INTEL_CONFIDENCE</span>
                  <div className="text-[20px] font-medium text-[24px] leading-none tracking-[-0.05em]">97.4%</div>
                </div>
                <div className="p-3 border border-[#3b494b]/30">
                  <span className="text-[11px] font-bold tracking-[0.1em] text-[#b9cacb] block mb-1">RISK_PARAMETER</span>
                  <div className="text-[20px] font-medium text-[24px] leading-none tracking-[-0.05em] text-[#ffb4ab]">NOMINAL</div>
                </div>
              </div>
              <div className="relative group">
                <input
                  className="w-full bg-[#353534]/10 border border-[#3b494b]/50 p-4 text-[14px] leading-relaxed focus:border-[#00dbe9] focus:ring-0 placeholder:text-[#b9cacb]/30"
                  placeholder="EXECUTE COMMAND_PROMPT..."
                  type="text"
                />
                <span className="absolute right-4 top-1/2 -translate-y-1/2 material-symbols-outlined text-[#00dbe9]">keyboard_return</span>
              </div>
            </div>
          </section>

          {/* Live Intelligence Feed */}
          <section className="col-span-12 md:col-span-5 glass-panel flex flex-col h-[400px]">
            <div className="p-4 border-b border-[#3b494b]/30 flex justify-between items-center">
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#b9cacb]">LIVE_TACTICAL_FEED</span>
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#00dbe9]">REAL_TIME</span>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3 text-[14px] leading-relaxed text-[12px]">
              <div className="flex gap-3">
                <span className="text-[#b9cacb]/40">[14:22:01]</span>
                <span className="text-[#00e639]">UPLINK_SUCCESS</span>
                <span className="text-[#b9cacb]">Node #NA-01 stabilized.</span>
              </div>
              <div className="flex gap-3">
                <span className="text-[#b9cacb]/40">[14:22:04]</span>
                <span className="text-[#00dbe9]">INTEL_SYNTH</span>
                <span className="text-[#b9cacb]">Neural cluster Alpha-9 completed cycle.</span>
              </div>
              <div className="flex gap-3">
                <span className="text-[#b9cacb]/40">[14:22:12]</span>
                <span className="text-[#ffb4ab]">LATENCY_SPIKE</span>
                <span className="text-[#b9cacb]">APAC routing delayed by 12ms.</span>
              </div>
              <div className="flex gap-3">
                <span className="text-[#b9cacb]/40">[14:22:18]</span>
                <span className="text-[#00e639]">AUTO_REROUTE</span>
                <span className="text-[#b9cacb]">Traffic shifted to EU central.</span>
              </div>
              <div className="flex gap-3">
                <span className="text-[#b9cacb]/40">[14:22:25]</span>
                <span className="text-[#00dbe9]">INTEL_SYNTH</span>
                <span className="text-[#b9cacb]">Batch #409-Z processing starts...</span>
              </div>
              <div className="flex gap-3">
                <span className="text-[#b9cacb]/40">[14:22:33]</span>
                <span className="text-[#b9cacb]">Awaiting system prompt...</span>
              </div>
            </div>
            <div className="p-2 border-t border-[#3b494b]/30 bg-[#1c1b1b]/50">
              <div className="flex justify-center items-center gap-4 py-2">
                <div className="w-1 h-1 bg-[#00dbe9] rounded-full animate-ping"></div>
                <span className="text-[11px] font-bold tracking-[0.1em] text-[#00dbe9] tracking-[0.3em]">PROCESSING DATA STREAM</span>
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* BottomNavBar (Mobile Only) */}
      <nav className="md:hidden fixed bottom-0 left-0 w-full z-50 flex justify-around items-stretch h-16 bg-[#0e0e0e]/90 backdrop-blur-md border-t border-[#3b494b]/50">
        <a className="flex flex-col items-center justify-center text-[#00dbe9] bg-[#00f0ff]/20 border-t-2 border-[#00dbe9] py-2 px-4 cursor-pointer">
          <span className="material-symbols-outlined">grid_view</span>
          <span className="text-[11px] font-bold tracking-[0.1em]">DASHBOARD</span>
        </a>
        <a className="flex flex-col items-center justify-center text-[#b9cacb]/60 py-2 px-4 hover:text-[#00dbe9] hover:bg-[#353534]/30 cursor-pointer" onClick={() => navigate('/labs')}>
          <span className="material-symbols-outlined">science</span>
          <span className="text-[11px] font-bold tracking-[0.1em]">UNITS</span>
        </a>
        <a className="flex flex-col items-center justify-center text-[#b9cacb]/60 py-2 px-4 hover:text-[#00dbe9] hover:bg-[#353534]/30 cursor-pointer">
          <span className="material-symbols-outlined">assignment_late</span>
          <span className="text-[11px] font-bold tracking-[0.1em]">MISSION</span>
        </a>
        <a className="flex flex-col items-center justify-center text-[#b9cacb]/60 py-2 px-4 hover:text-[#00dbe9] hover:bg-[#353534]/30 cursor-pointer">
          <span className="material-symbols-outlined">settings_input_component</span>
          <span className="text-[11px] font-bold tracking-[0.1em]">SYSTEM</span>
        </a>
      </nav>
    </div>
  );
}
