import { Link, useNavigate } from 'react-router';
import { APP_BOTTOM_NAV_OS } from '../../lib/app-nav';
import { Image3DViewer } from '../components/Image3DViewer';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function MetabolicOptimizerLab() {
  const navigate = useNavigate();
  const labHealth = useLabHealth('metabolic');

  return (
    <div className="min-h-screen qulab-page-bg text-foreground font-['JetBrains_Mono']">
      <style>{`
        .segmented-bar div {
          height: 100%;
          border-right: 2px solid #131313;
        }
      `}</style>

      {/* Top Navigation */}
      <header className="flex justify-between items-center w-full px-4 md:px-8 py-2 border-b border-[#3b494b]/50 bg-[#131313]/80 backdrop-blur-xl fixed top-0 z-50">
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-[#00dbe9]">terminal</span>
          <span className="font-['Space_Grotesk'] text-xl font-bold text-[#00dbe9] tracking-tighter uppercase">QULAB_INF_OS // V.1.0.4</span>
        </div>
        <div className="hidden md:flex items-center gap-4">
          <div className="flex flex-col items-end">
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#b9cacb]/60">SYSTEM_STATUS</span>
            <span className="font-['JetBrains_Mono'] text-2xl tracking-[-0.05em] font-medium text-[#00e639]">OPTIMIZED</span>
          </div>
          <button className="px-4 py-2 border border-[#00dbe9] text-[#00dbe9] font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold hover:bg-[#00dbe9]/10 transition-colors active:scale-95 duration-75 beveled-edge">
            [ENCRYPTED]
          </button>
        </div>
      </header>

      {/* Side Navigation (Desktop Only) */}
      <aside className="hidden md:flex h-full w-80 fixed left-0 top-0 bg-[#2a2a2a]/80 backdrop-blur-2xl border-r border-[#3b494b]/20 flex-col p-4 space-y-2 pt-24 z-40">
        <div className="pb-4 border-b border-[#3b494b]/20">
          <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00e639] uppercase">ECHO_INTEL_OVERLAY</span>
        </div>
        <nav className="flex-1 space-y-2">
          <button
            onClick={() => navigate('/labs')}
            className="w-full flex items-center gap-4 p-3 bg-[#13ff43]/10 text-[#72ff70] font-bold border-l-4 border-[#72ff70] hover:bg-[#e5e2e1]/5 transition-all"
          >
            <span className="material-symbols-outlined">settings_ethernet</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold">STREAM_01</span>
          </button>
          <Link to="/agent-telemetry" className="flex items-center gap-4 p-3 text-[#e5e2e1]/50 hover:bg-[#e5e2e1]/5 transition-all">
            <span className="material-symbols-outlined">analytics</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold">TELEMETRY</span>
          </Link>
          <Link to="/synthesis-archive" className="flex items-center gap-4 p-3 text-[#e5e2e1]/50 hover:bg-[#e5e2e1]/5 transition-all">
            <span className="material-symbols-outlined">history_edu</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold">COMMS_LOG</span>
          </Link>
          <Link to="/echo-mission" className="flex items-center gap-4 p-3 text-[#e5e2e1]/50 hover:bg-[#e5e2e1]/5 transition-all">
            <span className="material-symbols-outlined">psychology</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold">NEURAL_MAP</span>
          </Link>
        </nav>
        <div className="glass-panel p-4 beveled-edge">
          <div className="flex justify-between items-center mb-2">
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#b9cacb]">SEC_PROTOCOL</span>
            <span className="text-[#00e639] font-bold">NIST-T</span>
          </div>
          <div className="h-1 bg-[#353534] w-full">
            <div className="h-full bg-[#00e639] w-[88%]"></div>
          </div>
        </div>
      </aside>

      {/* Main Content Canvas */}
      <main className="md:ml-80 pt-24 pb-20 px-4 md:px-8 min-h-screen">
        <EchoLabCommandInline className="mb-6 max-w-7xl mx-auto" />
        <div className="grid grid-cols-12 gap-2 max-w-7xl mx-auto">
          {/* Header Row */}
          <div className="col-span-12 flex justify-between items-end mb-4 border-b border-[#3b494b]/30 pb-4">
            <div>
              <h1 className="font-['Space_Grotesk'] text-5xl font-bold text-[#e5e2e1] uppercase tracking-tight leading-[1.1]">Cancer Metabolic Optimizer</h1>
              <p className="font-['JetBrains_Mono'] text-sm text-[#b9cacb]">PORT_8001 // TREATMENT_REFINEMENT_ENGINE</p>
            </div>
            <div className="text-right">
              <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00dbe9]">ACTIVE_MISSION</span>
              <p className="font-['Space_Grotesk'] text-xl font-semibold text-[#00dbe9]">STG-3_OVARIAN_OPT</p>
            </div>
          </div>

          {/* Tumor Optimization Visualizer (Bento Main) */}
          <section className="col-span-12 lg:col-span-8 glass-panel p-4 beveled-edge min-h-[400px] flex flex-col relative overflow-hidden">
            <div className="flex justify-between items-start z-10">
              <div className="flex flex-col">
                <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#b9cacb]">TUMOR_OPTIMIZATION_VISUALIZER</span>
                <span className="font-['JetBrains_Mono'] text-sm text-[#00dbe9]">10-FIELD PROBABILITY CLOUDS</span>
              </div>
              <div className="flex gap-2">
                <span className="w-2 h-2 rounded-full bg-[#00dbe9] animate-pulse"></span>
                <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold">REALTIME_SCAN</span>
              </div>
            </div>

            {/* Probability Cloud Visualization */}
            <div className="flex-1 flex items-center justify-center relative">
              <div className="absolute inset-0 opacity-20 pointer-events-none">
                <div className="w-full h-full" style={{ backgroundImage: 'radial-gradient(circle at 50% 50%, #00dbe9 0.5px, transparent 0.5px)', backgroundSize: '24px 24px' }}></div>
              </div>
              <Image3DViewer
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuCfABQgPSeppcpmdvcvmH-MoooUuwcs80Udr31yVt4rVzYXq9DBfgqZz4rPwG7pVPtbXKhxG7OoxvDy9LWLQ45FKN4SgLGll-jGMH79k63gLoJDQdH7D35z1QFLq7fY70_ref05fclVgUyDoiKH-ODKjrQ4lLsBJbGFq2rDowKuDaEWCSkYabdISra4QTFy95LCN0vHDGfb9K0bC1dOfD47pBVEMGTDeHpSmzzk8DeFePTpnLaO2dCGWkXl7QokE18wxSPL7ngLDO8"
                alt="A sophisticated medical visualization representing 10-field tumor optimization as complex, glowing probability clouds in electric cyan and emerald green"
                className="w-full h-full object-cover mix-blend-screen opacity-60 rounded"
                autoRotate={true}
              />
              {/* Overlay Grid */}
              <div className="absolute inset-0 grid grid-cols-10 grid-rows-1 gap-1 pointer-events-none p-4">
                <div className="border-x border-white/5 flex items-end pb-2 justify-center font-['JetBrains_Mono'] text-[8px] tracking-[0.1em] font-bold text-[#00dbe9]/50">FLD_01</div>
                <div className="border-x border-white/5 flex items-end pb-2 justify-center font-['JetBrains_Mono'] text-[8px] tracking-[0.1em] font-bold text-[#00dbe9]/50">FLD_02</div>
                <div className="border-x border-white/5 flex items-end pb-2 justify-center font-['JetBrains_Mono'] text-[8px] tracking-[0.1em] font-bold text-[#00dbe9]/50">FLD_03</div>
                <div className="border-x border-[#00dbe9]/20 bg-[#00dbe9]/5 flex items-end pb-2 justify-center font-['JetBrains_Mono'] text-[8px] tracking-[0.1em] font-bold text-[#00dbe9]">FLD_04</div>
                <div className="border-x border-white/5 flex items-end pb-2 justify-center font-['JetBrains_Mono'] text-[8px] tracking-[0.1em] font-bold text-[#00dbe9]/50">FLD_05</div>
                <div className="border-x border-white/5 flex items-end pb-2 justify-center font-['JetBrains_Mono'] text-[8px] tracking-[0.1em] font-bold text-[#00dbe9]/50">FLD_06</div>
                <div className="border-x border-white/5 flex items-end pb-2 justify-center font-['JetBrains_Mono'] text-[8px] tracking-[0.1em] font-bold text-[#00dbe9]/50">FLD_07</div>
                <div className="border-x border-white/5 flex items-end pb-2 justify-center font-['JetBrains_Mono'] text-[8px] tracking-[0.1em] font-bold text-[#00dbe9]/50">FLD_08</div>
                <div className="border-x border-white/5 flex items-end pb-2 justify-center font-['JetBrains_Mono'] text-[8px] tracking-[0.1em] font-bold text-[#00dbe9]/50">FLD_09</div>
                <div className="border-x border-white/5 flex items-end pb-2 justify-center font-['JetBrains_Mono'] text-[8px] tracking-[0.1em] font-bold text-[#00dbe9]/50">FLD_10</div>
              </div>
            </div>

            <div className="mt-4 flex gap-4 overflow-x-auto pb-2">
              <div className="min-w-[120px] glass-panel p-2 border-l-2 border-[#00dbe9]">
                <span className="block font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#b9cacb]">CONVERGENCE</span>
                <span className="font-['JetBrains_Mono'] text-2xl tracking-[-0.05em] font-medium text-[#00dbe9]">0.9928</span>
              </div>
              <div className="min-w-[120px] glass-panel p-2 border-l-2 border-[#00e639]">
                <span className="block font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#b9cacb]">ENTROPY_GAP</span>
                <span className="font-['JetBrains_Mono'] text-2xl tracking-[-0.05em] font-medium text-[#00e639]">0.002</span>
              </div>
              <div className="min-w-[120px] glass-panel p-2 border-l-2 border-[#ffb4ab]">
                <span className="block font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#b9cacb]">LATENCY</span>
                <span className="font-['JetBrains_Mono'] text-2xl tracking-[-0.05em] font-medium text-[#ffb4ab]">12ms</span>
              </div>
            </div>
          </section>

          {/* Real-time Telemetry (Side Stack) */}
          <section className="col-span-12 lg:col-span-4 space-y-2">
            <div className="glass-panel p-4 beveled-edge">
              <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#b9cacb] mb-4 block">METABOLIC_RATE_FEED</span>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold mb-1">
                    <span>GLUCOSE_UPTAKE</span>
                    <span className="text-[#00dbe9]">92%</span>
                  </div>
                  <div className="h-4 w-full bg-[#353534]/30 segmented-bar flex gap-[1px]">
                    <div className="bg-[#00dbe9] flex-1"></div>
                    <div className="bg-[#00dbe9] flex-1"></div>
                    <div className="bg-[#00dbe9] flex-1"></div>
                    <div className="bg-[#00dbe9] flex-1"></div>
                    <div className="bg-[#00dbe9] flex-1"></div>
                    <div className="bg-[#00dbe9]/20 flex-1"></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold mb-1">
                    <span>LACTATE_EFFLUX</span>
                    <span className="text-[#00e639]">44%</span>
                  </div>
                  <div className="h-4 w-full bg-[#353534]/30 segmented-bar flex gap-[1px]">
                    <div className="bg-[#00e639] flex-1"></div>
                    <div className="bg-[#00e639] flex-1"></div>
                    <div className="bg-[#00e639]/20 flex-1"></div>
                    <div className="bg-[#00e639]/20 flex-1"></div>
                    <div className="bg-[#00e639]/20 flex-1"></div>
                    <div className="bg-[#00e639]/20 flex-1"></div>
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-panel p-4 beveled-edge">
              <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#b9cacb] mb-4 block">KILL_RATE_BENCHMARKS</span>
              <div className="grid grid-cols-2 gap-2">
                <div className="p-3 bg-[#0e0e0e] border border-[#3b494b]/30 text-center">
                  <span className="block font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#b9cacb]">EST_CLEARANCE</span>
                  <span className="font-['JetBrains_Mono'] text-2xl tracking-[-0.05em] font-medium text-[#00dbe9]">78.4%</span>
                </div>
                <div className="p-3 bg-[#0e0e0e] border border-[#3b494b]/30 text-center">
                  <span className="block font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#b9cacb]">PROB_ESCAPE</span>
                  <span className="font-['JetBrains_Mono'] text-2xl tracking-[-0.05em] font-medium text-[#ffb4ab]">2.1%</span>
                </div>
              </div>
            </div>

            <div className="glass-panel p-4 beveled-edge border-l-4 border-[#00dbe9] bg-[#00f0ff]/5">
              <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00dbe9] mb-2 block">QUICK_ACTION_CMD</span>
              <button className="w-full py-4 bg-[#00dbe9] text-[#00363a] font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold active-glow transition-all active:scale-95 flex items-center justify-center gap-3">
                <span className="material-symbols-outlined">rocket_launch</span>
                OPTIMIZE TREATMENT FOR STAGE 3 OVARIAN CANCER
              </button>
            </div>
          </section>

          {/* Echo Intelligence & Grounded Reasoning */}
          <section className="col-span-12 md:col-span-7 glass-panel p-4 beveled-edge min-h-[300px]">
            <div className="flex items-center gap-3 mb-6 border-b border-[#3b494b]/30 pb-4">
              <span className="material-symbols-outlined text-[#00e639]">psychology</span>
              <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#e5e2e1]">ECHO_INTEL_SYSTEM // GROUNDED_REASONING</span>
            </div>
            <div className="space-y-4 font-['JetBrains_Mono'] text-sm text-[#b9cacb] leading-relaxed">
              <p className="flex gap-4">
                <span className="text-[#00e639] shrink-0">[00:12:44]</span>
                <span>Analysis of metabolic flux indicates high dependency on aerobic glycolysis within the primary tumor volume. Field convergence has been shifted by 4.2mm to intercept the peripheral oxygenation supply chain.</span>
              </p>
              <p className="flex gap-4 p-3 bg-[#0e0e0e] border border-[#3b494b]/10">
                <span className="text-[#00dbe9] shrink-0">REASONING:</span>
                <span>Stage 3 Ovarian pathology suggests increased resistance to standard field parameters. System-level optimization suggests a dual-phased pulse loop to destabilize Mitochondrial ATP production before terminal field application.</span>
              </p>
              <div className="p-4 border border-dashed border-[#3b494b]/50 flex flex-col gap-2">
                <span className="font-['JetBrains_Mono'] text-[10px] tracking-[0.1em] font-bold text-[#00e639]">OPTIMIZATION_SUGGESTION_ID: 994-B</span>
                <div className="flex items-center justify-between">
                  <span className="text-[#e5e2e1] italic">"Increase Field 4 Intensity by 18% to counteract necrotic core pressure."</span>
                  <button className="text-[#00dbe9] font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold border border-[#00dbe9] px-3 py-1 hover:bg-[#00dbe9]/10">EXECUTE</button>
                </div>
              </div>
            </div>
          </section>

          {/* Logs & Security Status */}
          <section className="col-span-12 md:col-span-5 glass-panel p-4 beveled-edge">
            <div className="flex justify-between items-center mb-4">
              <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#b9cacb]">NIST-TRACEABLE_LOGS</span>
              <span className="material-symbols-outlined text-[#b9cacb]/40">lock</span>
            </div>
            <div className="bg-[#0e0e0e]/50 p-2 font-mono text-[11px] h-[200px] overflow-y-auto space-y-1">
              <div className="text-[#b9cacb]/60"><span className="text-[#00e639]">SEC_OK</span> // 14:22:01 // NIST_P-8001 Verified</div>
              <div className="text-[#b9cacb]/60"><span className="text-[#00e639]">LOG_IN</span> // 14:22:04 // Treatment_Vector_Calc_Init</div>
              <div className="text-[#00dbe9]"><span className="text-[#00dbe9]">CMD_EX</span> // 14:22:09 // Metabolic_Probe_Active</div>
              <div className="text-[#b9cacb]/60"><span className="text-[#00e639]">SEC_OK</span> // 14:22:15 // Hash_Match_Success_3F92X</div>
              <div className="text-[#ffb4ab]"><span className="text-[#ffb4ab]">ALRT_W</span> // 14:22:22 // Thermogenic_Variance_Detected</div>
              <div className="text-[#b9cacb]/60"><span className="text-[#00e639]">LOG_IN</span> // 14:22:30 // Re-routing_Aux_Compute</div>
              <div className="text-[#b9cacb]/60"><span className="text-[#00e639]">SEC_OK</span> // 14:22:45 // Integrity_Check_100%</div>
            </div>
            <div className="mt-4 pt-4 border-t border-[#3b494b]/30 grid grid-cols-2 gap-4">
              <div className="flex flex-col">
                <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#b9cacb]">IDENTITY_STATE</span>
                <span className="text-[#00dbe9] font-bold">VERIFIED</span>
              </div>
              <div className="flex flex-col">
                <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#b9cacb]">ENCRYPTION_LEVEL</span>
                <span className="text-[#e5e2e1] font-bold">AES-512_TAC</span>
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* Bottom Navigation (Mobile Only) */}
      <footer className="md:hidden fixed bottom-0 left-0 w-full z-50 flex justify-around items-stretch h-16 bg-[#0e0e0e]/90 backdrop-blur-md border-t border-[#3b494b]/50">
        <button
          onClick={() => navigate('/labs')}
          className="flex flex-col items-center justify-center text-[#00dbe9] bg-[#00f0ff]/20 border-t-2 border-[#00dbe9] py-2 px-4 transition-all"
        >
          <span className="material-symbols-outlined">grid_view</span>
          <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold mt-1">DASHBOARD</span>
        </button>
        {APP_BOTTOM_NAV_OS.slice(1).map((item) => (
          <button
            key={item.path}
            type="button"
            onClick={() => navigate(item.path)}
            className="flex flex-col items-center justify-center text-[#b9cacb]/60 py-2 px-4 hover:text-[#00dbe9] hover:bg-[#353534]/30"
          >
            <span className="material-symbols-outlined">{item.icon}</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold mt-1">{item.label.toUpperCase()}</span>
          </button>
        ))}
      </footer>
    </div>
  );
}
