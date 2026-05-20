import { useState } from 'react';
import { useNavigate } from 'react-router';
import { apiClient } from '../../lib/api-client';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function SynthesisArchive() {
  const navigate = useNavigate();
  const { health, loading: healthLoading } = useLabHealth('global');
  const [searchQuery, setSearchQuery] = useState('');
  const [materialFilter, setMaterialFilter] = useState('ALL');
  const [successFilter, setSuccessFilter] = useState('ALL');

  const archiveRecords = [
    {
      id: 'QN-902-DELTA',
      status: 'VERIFIED',
      materialType: 'CRYSTAL LATTICE',
      outcome: 'High-tensile superconducting mesh with 99.8% stability at 14K.',
      steps: [
        { num: '01', text: 'Injected Ar-plasma substrate at 1200Pa. Stabilize thermal gradient within 0.05ms.' },
        { num: '02', text: 'Deploy nanite-swarms for molecular lattice alignment. Phase shift: 45°.', highlight: true },
        { num: '03', text: 'Flash cooling via cryogenic pulse. Verify structural integrity 1ms post-flash.' },
      ],
      troubleshooting: 'Lattice fracture detected in 0.4% of iterations. Solution: Increase Ar-flow by 2%.',
      verification: 99.8,
    },
    {
      id: 'MX-114-SIGMA',
      status: 'STABLE',
      materialType: 'BIO-POLYMERIC',
      outcome: 'Self-healing synthetic tissue for deep-space ocular shielding.',
      steps: [
        { num: '01', text: 'Culture synthetic ocular cells in oxygenated nutrient vat. PH 7.4.' },
        { num: '02', text: 'Introduce nanite weavers to bind protein chains. Temp 37.2C.', highlight: true },
        { num: '03', text: 'UV curing sequence: 250nm for 120s. Monitor for chain collapse.' },
      ],
      troubleshooting: 'UV overexposure leads to brittleness. Calibration: Reduce pulse by 5s.',
      verification: 87.5,
    },
  ];

  return (
    <>
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #131313;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #3b494b;
        }
      `}</style>

      <div className="min-h-screen qulab-page-bg text-foreground font-['JetBrains_Mono'] overflow-hidden flex flex-col">
        {/* TopAppBar */}
        <header className="bg-[#131313]/80 backdrop-blur-xl border-b border-[#3b494b] flex justify-between items-center px-8 w-full z-50 fixed top-0 h-16">
          <div className="flex items-center gap-4">
            <span className="material-symbols-outlined text-[#dbfcff]">biotech</span>
            <h1 className="text-xl font-bold tracking-tighter text-[#dbfcff] uppercase">QULAB INFINITE // SYSTEM v2.4</h1>
          </div>
          <div className="hidden md:flex items-center gap-6">
            <nav className="flex gap-6">
              <button onClick={() => navigate('/')} className="text-[#b9cacb] hover:bg-[#353534]/30 transition-colors px-2 py-1 text-[11px] font-bold tracking-[0.1em]">DASHBOARD</button>
              <button className="text-[#dbfcff] font-bold hover:bg-[#353534]/30 transition-colors px-2 py-1 text-[11px] tracking-[0.1em]">ARCHIVE</button>
              <button className="text-[#b9cacb] hover:bg-[#353534]/30 transition-colors px-2 py-1 text-[11px] font-bold tracking-[0.1em]">PROTOCOLS</button>
            </nav>
            <button className="material-symbols-outlined text-[#b9cacb] hover:text-[#dbfcff] transition-colors">settings_input_component</button>
          </div>
        </header>

        <div className="flex flex-1 pt-16 h-full overflow-hidden">
          {/* NavigationDrawer (Desktop) */}
          <aside className="bg-[#0e0e0e]/90 backdrop-blur-lg border-r border-[#3b494b] hidden md:flex flex-col py-8 px-4 h-screen w-64 shrink-0">
            <div className="mb-8">
              <p className="text-[11px] text-[#dbfcff] mb-2 font-bold tracking-[0.1em]">CORE TERMINAL</p>
              <div className="h-[1px] bg-[#3b494b] w-full"></div>
            </div>
            <nav className="flex flex-col gap-2">
              <button onClick={() => navigate('/materials-lab')} className="flex items-center gap-3 p-3 text-[#b9cacb] hover:bg-[#2a2a2a] transition-all text-[11px] font-bold tracking-[0.1em]">
                <span className="material-symbols-outlined">visibility</span>
                Active Visualizer
              </button>
              <button className="flex items-center gap-3 p-3 text-[#b9cacb] hover:bg-[#2a2a2a] transition-all text-[11px] font-bold tracking-[0.1em]">
                <span className="material-symbols-outlined">grid_view</span>
                Parameter Matrix
              </button>
              <button className="flex items-center gap-3 p-3 text-[#b9cacb] hover:bg-[#2a2a2a] transition-all text-[11px] font-bold tracking-[0.1em]">
                <span className="material-symbols-outlined">psychology</span>
                Echo Insights
              </button>
              <button className="flex items-center gap-3 p-3 bg-[#13ff43]/10 text-[#007117] rounded-none border-l-4 border-[#13ff43] transition-all text-[11px] font-bold tracking-[0.1em]">
                <span className="material-symbols-outlined">database</span>
                Telemetry Logs
              </button>
            </nav>
            <div className="mt-auto pt-8">
              <button onClick={() => navigate('/live-lab-wall')} className="flex items-center justify-between group p-3 border border-[#3b494b] hover:border-[#dbfcff] transition-colors w-full">
                <span className="text-[11px] text-[#b9cacb] group-hover:text-[#dbfcff] transition-colors font-bold tracking-[0.1em]">LIVE LAB ACCESS</span>
                <span className="material-symbols-outlined text-[#dbfcff]">bolt</span>
              </button>
            </div>
          </aside>

          {/* Main Content Canvas */}
                <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="flex-1 flex flex-col h-full overflow-hidden bg-[#131313]">
            {/* Filter Bar */}
            <section className="p-8 border-b border-[#3b494b] flex flex-col md:flex-row md:items-end justify-between gap-4">
              <div className="space-y-2">
                <h2 className="text-3xl text-[#dbfcff] tracking-tight font-semibold">SYNTHESIS ARCHIVE</h2>
                <p className="text-[#b9cacb] text-sm max-w-2xl">Reference-grade library of validated protocols and material outcomes generated by the Echo probabilistic engine.</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <div className="relative">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 material-symbols-outlined text-[#849495] text-[18px]">search</span>
                  <input
                    className="bg-white/5 border border-[#3b494b] focus:border-[#dbfcff] focus:ring-0 text-[#e5e2e1] px-10 py-2 text-[11px] w-64 transition-colors font-bold tracking-[0.1em]"
                    placeholder="SEARCH IDS OR MATERIALS..."
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                </div>
                <select
                  className="bg-white/5 border border-[#3b494b] text-[#e5e2e1] text-[11px] py-2 px-4 focus:ring-0 appearance-none font-bold tracking-[0.1em]"
                  value={materialFilter}
                  onChange={(e) => setMaterialFilter(e.target.value)}
                >
                  <option>MATERIAL TYPE: ALL</option>
                  <option>POLYMER</option>
                  <option>G-ALLOY</option>
                  <option>QUANTUM FLUID</option>
                </select>
                <select
                  className="bg-white/5 border border-[#3b494b] text-[#e5e2e1] text-[11px] py-2 px-4 focus:ring-0 appearance-none font-bold tracking-[0.1em]"
                  value={successFilter}
                  onChange={(e) => setSuccessFilter(e.target.value)}
                >
                  <option>SUCCESS LEVEL: ALL</option>
                  <option>VERIFIED (100%)</option>
                  <option>STABLE (85%+)</option>
                </select>
              </div>
            </section>

            {/* Scrollable Grid */}
            <section className="flex-1 overflow-y-auto p-8 custom-scrollbar">
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                {archiveRecords.map((record) => (
                  <div key={record.id} className="tactical-glass p-6 flex flex-col gap-6 relative group hover:active-glow transition-all">
                    <div className="flex justify-between items-start">
                      <div className="space-y-1">
                        <span className="text-[11px] text-[#849495] font-bold tracking-[0.1em]">SYNTHESIS ID</span>
                        <div className="text-2xl text-[#dbfcff] font-medium tracking-tight">{record.id}</div>
                      </div>
                      <div className={`flex items-center gap-2 px-3 py-1 text-[11px] font-bold tracking-[0.1em] ${
                        record.status === 'VERIFIED'
                          ? 'bg-[#72ff70] text-[#002203]'
                          : 'bg-[#2a2a2a] text-[#b9cacb] border border-[#3b494b]'
                      }`}>
                        {record.status === 'VERIFIED' && (
                          <span className="material-symbols-outlined text-[14px]" style={{ fontVariationSettings: '"FILL" 1' }}>verified</span>
                        )}
                        {record.status === 'VERIFIED' ? 'VERIFIED' : 'STABLE'}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 border-y border-[#3b494b]/30 py-4">
                      <div className="space-y-1">
                        <span className="text-[11px] text-[#849495] font-bold tracking-[0.1em]">MATERIAL TYPE</span>
                        <div className="text-[#e5e2e1]">{record.materialType}</div>
                      </div>
                      <div className="space-y-1 md:col-span-2">
                        <span className="text-[11px] text-[#849495] font-bold tracking-[0.1em]">GROUNDED OUTCOME</span>
                        <div className="text-[#e5e2e1]">{record.outcome}</div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h4 className="text-[11px] text-[#dbfcff] flex items-center gap-2 font-bold tracking-[0.1em]">
                        <span className="material-symbols-outlined text-[16px]">account_tree</span>
                        ECHO'S STEP-BY-STEP PROTOCOL
                      </h4>
                      <div className="space-y-3">
                        {record.steps.map((step) => (
                          <div key={step.num} className="flex gap-4 items-start">
                            <div className={`shrink-0 w-6 h-6 border flex items-center justify-center text-[10px] font-bold tracking-[0.1em] ${
                              step.highlight ? 'border-[#dbfcff] text-[#dbfcff]' : 'border-[#849495] text-[#849495]'
                            }`}>{step.num}</div>
                            <p className="text-[#b9cacb] text-sm">{step.text}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="p-3 bg-[#93000a]/10 border border-[#ffb4ab]/20 space-y-2">
                        <div className="text-[11px] text-[#ffb4ab] flex items-center gap-2 font-bold tracking-[0.1em]">
                          <span className="material-symbols-outlined text-[14px]">report_problem</span>
                          TROUBLESHOOTING LOG
                        </div>
                        <p className="text-[#b9cacb] text-[12px]">{record.troubleshooting}</p>
                      </div>
                      <div className="p-3 bg-[#13ff43]/5 border border-[#72ff70]/20 space-y-2">
                        <div className="text-[11px] text-[#72ff70] flex items-center gap-2 font-bold tracking-[0.1em]">
                          <span className="material-symbols-outlined text-[14px]">fact_check</span>
                          MATERIAL VERIFICATION
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-2 bg-[#2a2a2a] rounded-none overflow-hidden flex gap-[2px]">
                            <div className="h-full bg-[#72ff70]" style={{ width: `${record.verification}%` }}></div>
                          </div>
                          <span className="text-[10px] text-[#72ff70] font-bold tracking-[0.1em]">{record.verification}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}

                {/* Visualization Card */}
                <div className="tactical-glass overflow-hidden h-full flex flex-col min-h-[400px] xl:col-span-2">
                  <div className="p-4 border-b border-[#3b494b] flex justify-between items-center bg-[#1c1b1b]">
                    <span className="text-[11px] text-[#dbfcff] font-bold tracking-[0.1em]">MICROSCOPIC LATTICE VISUALIZATION // RUN_661</span>
                    <span className="material-symbols-outlined text-[#849495]">fullscreen</span>
                  </div>
                  <div className="flex-1 relative bg-black">
                    <img
                      alt="Synthesis Visualization"
                      className="w-full h-full object-cover opacity-60 grayscale hover:grayscale-0 transition-all duration-700"
                      src="https://lh3.googleusercontent.com/aida-public/AB6AXuDCSKFc1AaqyHwt-Xa89tVxxZAFg6qY0ZmeqQTp_oxeWgvRMeptRbmWT28pFrLv5j76NieSgKfm_V5VK6fJHPcIxmx0Ioy19EH6uUQMFNRTlqNHcQnKXKTTb7R8MmaZRKwSNDznXaGLSwy7fY029tJ82bssMDOAJ-_gukD89Lz5XR9XXANvvtw6sbUu8VQf9tswLMhawjL4gFGR3mnVWIZ51-sr-UCI7CJaJs5oW3J7W3-ajyyNuFLic2S2M8sBCMQzv_MgGeBAyMo"
                    />
                    <div className="absolute inset-0 pointer-events-none border-[16px] border-black/20"></div>
                    <div className="absolute top-8 left-8 p-4 bg-black/60 backdrop-blur-md border border-[#dbfcff]/30">
                      <p className="text-[11px] text-[#dbfcff] mb-2 font-bold tracking-[0.1em]">LIVE SCAN PARAMETERS</p>
                      <ul className="space-y-1 text-[10px] text-[#b9cacb] font-bold tracking-[0.1em]">
                        <li>RESOLUTION: 0.02nm</li>
                        <li>PHASE: ALIGNED</li>
                        <li>THREAT: NONE</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </main>
        </div>

        {/* BottomNavBar (Mobile) */}
        <nav className="fixed bottom-0 left-0 w-full flex justify-around items-center h-16 bg-[#0e0e0e]/95 backdrop-blur-md border-t border-[#3b494b] md:hidden z-50">
          <button onClick={() => navigate('/materials-lab')} className="flex flex-col items-center justify-center text-[#b9cacb] p-2 text-[11px] font-bold tracking-[0.1em]">
            <span className="material-symbols-outlined">monitoring</span>
            VIEWPORT
          </button>
          <button className="flex flex-col items-center justify-center text-[#b9cacb] p-2 text-[11px] font-bold tracking-[0.1em]">
            <span className="material-symbols-outlined">science</span>
            SYNTHESIS
          </button>
          <button className="flex flex-col items-center justify-center text-[#dbfcff] bg-[#00f0ff]/10 p-2 text-[11px] font-bold tracking-[0.1em]">
            <span className="material-symbols-outlined">inventory_2</span>
            ARCHIVE
          </button>
          <button className="flex flex-col items-center justify-center text-[#b9cacb] p-2 text-[11px] font-bold tracking-[0.1em]">
            <span className="material-symbols-outlined">terminal</span>
            ECHO
          </button>
        </nav>
      </div>
    </>
  );
}
