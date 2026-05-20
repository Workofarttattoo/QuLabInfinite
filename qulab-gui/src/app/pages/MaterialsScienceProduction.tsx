import { useState } from 'react';
import { useNavigate } from 'react-router';
import { apiClient } from '../../lib/api-client';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function MaterialsScienceProduction() {
  const navigate = useNavigate();
  const { health, loading: healthLoading } = useLabHealth('global');

  const metrics = [
    { label: 'Lattice Stability', value: '0.9982', percentage: 100, color: 'primary' },
    { label: 'Purity Index', value: '99.9%', percentage: 87.5, color: 'success' },
    { label: 'Growth Rate', value: '12.4', unit: 'μm/s', percentage: 50, color: 'primary' },
    { label: 'Env Delta', value: '-0.002', percentage: 25, color: 'warning' },
  ];

  const echoInsights = [
    { time: '14:22:05', type: 'AUTO_GEN', text: 'Synthesis of the crystalline substrate has reached peak alignment. Lattice energy bonds show no signs of thermal decoupling.', highlight: true },
    { time: '14:18:22', type: 'MANUAL_LOG', text: 'Injected silicon dopant levels adjusted to 0.02%. Observing immediate stabilization in the northeast quadrant.', highlight: false },
    { time: '14:15:10', type: 'SYSTEM_CRIT', text: 'Environmental delta approaching safety threshold. Cooling systems engaged. Efficiency remains at 94%.', highlight: 'success' },
    { time: '14:02:45', type: 'ECHO_V2', text: 'Neural analysis suggests a 12% increase in growth velocity if pressure is increased by 0.5 ATM.', highlight: false },
  ];

  return (
    <>
      <style>{`
        .segmented-progress {
          display: flex;
          gap: 2px;
        }
        .progress-block {
          height: 8px;
          width: 100%;
          background: rgba(219, 252, 255, 0.1);
        }
        .progress-block.active {
          background: #00f0ff;
        }
        .progress-block.warning {
          background: #ffb4ab;
        }
        .scan-line {
          background: linear-gradient(to bottom, transparent 50%, rgba(219, 252, 255, 0.05) 50%);
          background-size: 100% 4px;
        }
        .cursor-flicker {
          width: 2px;
          height: 14px;
          background: #dbfcff;
          display: inline-block;
          margin-left: 2px;
          vertical-align: middle;
        }
      `}</style>

      <div className="min-h-screen qulab-page-bg text-foreground font-['JetBrains_Mono'] overflow-hidden flex flex-col">
        {/* TopAppBar */}
        <header className="bg-[#131313]/80 backdrop-blur-xl border-b border-[#3b494b] flex justify-between items-center px-8 w-full z-50 h-16 fixed top-0">
          <div className="flex items-center gap-4">
            <span className="material-symbols-outlined text-[#dbfcff]">biotech</span>
            <h1 className="text-xl font-bold tracking-tighter text-[#dbfcff] uppercase">QULAB INFINITE // SYSTEM v2.4</h1>
          </div>
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 px-3 py-1 border border-[#dbfcff]/20 bg-[#dbfcff]/5">
              <span className="w-2 h-2 rounded-full bg-[#72ff70] animate-pulse"></span>
              <span className="text-[11px] text-[#dbfcff] font-bold tracking-[0.1em]">REACTOR LIVE</span>
            </div>
            <button className="material-symbols-outlined text-[#b9cacb] hover:bg-[#353534]/30 transition-colors p-2">settings_input_component</button>
          </div>
        </header>

        {/* Side Navigation (Web) */}
        <aside className="hidden md:flex flex-col py-8 px-4 h-screen w-64 fixed left-0 top-0 mt-16 bg-[#0e0e0e]/90 backdrop-blur-lg border-r border-[#3b494b] z-40">
          <div className="mb-8">
            <h2 className="text-[11px] text-[#b9cacb] opacity-50 px-4 mb-4 font-bold tracking-[0.1em]">CORE TERMINAL</h2>
            <nav className="space-y-1">
              <button className="flex items-center gap-3 px-4 py-3 bg-[#13ff43]/10 text-[#007117] rounded-none border-l-4 border-[#13ff43] transition-all w-full text-left">
                <span className="material-symbols-outlined">visibility</span>
                <span className="text-[11px] font-bold tracking-[0.1em]">Active Visualizer</span>
              </button>
              <button className="flex items-center gap-3 px-4 py-3 text-[#b9cacb] hover:text-[#dbfcff] hover:bg-[#2a2a2a] transition-all w-full text-left">
                <span className="material-symbols-outlined">grid_view</span>
                <span className="text-[11px] font-bold tracking-[0.1em]">Parameter Matrix</span>
              </button>
              <button className="flex items-center gap-3 px-4 py-3 text-[#b9cacb] hover:text-[#dbfcff] hover:bg-[#2a2a2a] transition-all w-full text-left">
                <span className="material-symbols-outlined">psychology</span>
                <span className="text-[11px] font-bold tracking-[0.1em]">Echo Insights</span>
              </button>
              <button onClick={() => navigate('/synthesis-archive')} className="flex items-center gap-3 px-4 py-3 text-[#b9cacb] hover:text-[#dbfcff] hover:bg-[#2a2a2a] transition-all w-full text-left">
                <span className="material-symbols-outlined">database</span>
                <span className="text-[11px] font-bold tracking-[0.1em]">Telemetry Logs</span>
              </button>
            </nav>
          </div>
          <div className="mt-auto p-4 glass-panel border-dashed">
            <div className="flex justify-between mb-2">
              <span className="text-[10px] text-[#849495] font-bold tracking-[0.1em]">SYS_LOAD</span>
              <span className="text-[10px] text-[#dbfcff] font-bold tracking-[0.1em]">24.8%</span>
            </div>
            <div className="segmented-progress">
              <div className="progress-block active"></div>
              <div className="progress-block active"></div>
              <div className="progress-block active"></div>
              <div className="progress-block active"></div>
              <div className="progress-block"></div>
              <div className="progress-block"></div>
              <div className="progress-block"></div>
              <div className="progress-block"></div>
            </div>
          </div>
        </aside>

        {/* Main Content Canvas */}
              <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="flex-1 mt-16 md:ml-64 mb-16 md:mb-0 p-4 md:p-4 overflow-y-auto scan-line">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-2 h-full">
            {/* Hero Visualizer Block */}
            <div className="md:col-span-8 flex flex-col gap-2">
              <div className="glass-panel active-stroke flex-1 relative min-h-[400px] overflow-hidden">
                {/* Tile Header */}
                <div className="absolute top-4 left-4 z-10 flex flex-col gap-1">
                  <span className="text-[11px] text-[#dbfcff] opacity-70 font-bold tracking-[0.1em]">HEADER-LABEL: LATTICE_STREAM_01</span>
                  <h2 className="text-xl text-[#dbfcff] uppercase font-semibold">Material State Visualizer</h2>
                </div>
                <div className="absolute top-4 right-4 z-10 flex items-center gap-2 bg-[#dbfcff]/10 px-2 py-1 border border-[#dbfcff]/20">
                  <span className="w-2 h-2 bg-[#72ff70]"></span>
                  <span className="text-[10px] text-[#dbfcff] font-bold tracking-[0.1em]">STATUS: OPTIMAL</span>
                </div>
                {/* Primary Visualizer Image */}
                <div className="absolute inset-0 flex items-center justify-center p-8">
                  <img
                    alt="Complex 3D holographic crystal lattice growth structure"
                    className="w-full h-full object-contain opacity-90"
                    src="https://lh3.googleusercontent.com/aida/ADBb0ugO63Xz5R9_kQLZFJQ_wdeIMn3vKFAljVWvb-NR3u6_uZcXDwWtfKCrWLSkmztt17bET38cN3v733tlk6DrTtHZUF55NZBXG0ve894w_2MAiu0NISh7w6HvAX7dz_E367wPFahkUPb7dBxip4VWzWffyh1F1quvJrQRDcyrcctCcadjgDGazUaNtM89WLu6rTY-PaDSDsYy9ZeZ8yS3fdoKyGn_-l53llFz1Kvcg0da8jRnhLfaPKu1o60"
                  />
                </div>
                {/* Dynamic Overlays */}
                <div className="absolute bottom-4 left-4 z-10 flex gap-4">
                  <div className="bg-black/40 backdrop-blur-md p-3 border border-[#3b494b]">
                    <span className="text-[9px] block text-[#849495] mb-1 font-bold tracking-[0.1em]">ZOOM_LEVEL</span>
                    <span className="text-2xl text-[#dbfcff] font-medium tracking-tight">48.2x</span>
                  </div>
                  <div className="bg-black/40 backdrop-blur-md p-3 border border-[#3b494b]">
                    <span className="text-[9px] block text-[#849495] mb-1 font-bold tracking-[0.1em]">STABILITY_REF</span>
                    <span className="text-2xl text-[#72ff70] font-medium tracking-tight">99.4%</span>
                  </div>
                </div>
              </div>

              {/* Secondary Stats Row */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {metrics.map((metric) => (
                  <div key={metric.label} className="glass-panel p-4 flex flex-col justify-between h-32">
                    <span className="text-[11px] text-[#849495] uppercase font-bold tracking-[0.1em]">{metric.label}</span>
                    <div className="flex flex-col gap-2">
                      <span className={`text-2xl font-medium tracking-tight ${
                        metric.color === 'primary' ? 'text-[#dbfcff]' :
                        metric.color === 'success' ? 'text-[#72ff70]' :
                        'text-[#ffb4ab]'
                      }`}>
                        {metric.value}{metric.unit && <small className="text-sm">{metric.unit}</small>}
                      </span>
                      <div className="segmented-progress">
                        {Array.from({ length: 8 }).map((_, i) => (
                          <div
                            key={i}
                            className={`progress-block ${
                              i < (metric.percentage / 12.5) ? (metric.color === 'warning' ? 'warning' : 'active') : ''
                            }`}
                          ></div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Side Insights Panel */}
            <div className="md:col-span-4 flex flex-col gap-2">
              <div className="glass-panel flex-1 flex flex-col">
                <div className="p-4 border-b border-[#3b494b] flex justify-between items-center bg-[#2a2a2a]/30">
                  <div className="flex flex-col">
                    <span className="text-[10px] text-[#849495] font-bold tracking-[0.1em]">LOG_TYPE: QUANTUM_ECHO</span>
                    <h3 className="text-[11px] text-[#dbfcff] uppercase font-bold tracking-[0.1em]">Echo Insights</h3>
                  </div>
                  <span className="material-symbols-outlined text-[#dbfcff]">terminal</span>
                </div>
                <div className="p-4 flex-1 overflow-y-auto space-y-4">
                  {echoInsights.map((insight, i) => (
                    <div key={i} className={`border-l-2 pl-3 py-1 ${
                      insight.highlight === true ? 'border-[#dbfcff]' :
                      insight.highlight === 'success' ? 'border-[#72ff70] bg-[#72ff70]/5' :
                      'border-[#3b494b]'
                    }`}>
                      <span className={`text-[10px] block mb-1 font-bold tracking-[0.1em] ${
                        insight.highlight === true ? 'text-[#849495]' :
                        insight.highlight === 'success' ? 'text-[#72ff70]' :
                        'text-[#849495]'
                      }`}>{insight.time} // {insight.type}</span>
                      <p className={`text-xs leading-relaxed ${
                        insight.highlight === true ? 'text-[#dbfcff]' : 'text-[#b9cacb]'
                      }`}>{insight.text}</p>
                    </div>
                  ))}
                </div>
                <div className="p-4 border-t border-[#3b494b] bg-[#0e0e0e]">
                  <div className="flex items-center gap-2 px-3 py-2 bg-white/5 border border-[#3b494b]">
                    <span className="text-[#dbfcff] opacity-50 font-bold tracking-[0.1em]">&gt;</span>
                    <span className="flex-1 text-[12px] text-[#dbfcff] font-bold tracking-[0.1em]">Awaiting lab prompt...<span className="cursor-flicker"></span></span>
                  </div>
                </div>
              </div>

              {/* Secondary Insight Card */}
              <div className="glass-panel p-4 h-48 relative overflow-hidden group">
                <div className="absolute inset-0 z-0">
                  <img
                    alt="Abstract circuit board impurity map"
                    className="w-full h-full object-cover opacity-20 group-hover:opacity-40 transition-opacity"
                    src="https://www.gstatic.com/labs-code/stitch/stitch-placeholder-300x300.svg"
                  />
                </div>
                <div className="relative z-10 flex flex-col h-full justify-between">
                  <div className="flex justify-between items-start">
                    <span className="text-[11px] text-[#849495] uppercase font-bold tracking-[0.1em]">Impurity Map</span>
                    <span className="material-symbols-outlined text-[#72ff70]" style={{ fontVariationSettings: '"FILL" 1' }}>sensors</span>
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between">
                      <span className="text-[10px] text-[#849495] font-bold tracking-[0.1em]">SILICON_DOPANT</span>
                      <span className="text-[10px] text-[#dbfcff] font-bold tracking-[0.1em]">0.02%</span>
                    </div>
                    <div className="h-1 bg-white/10 w-full">
                      <div className="h-full bg-[#dbfcff]" style={{ width: '2%' }}></div>
                    </div>
                    <div className="flex justify-between mt-2">
                      <span className="text-[10px] text-[#849495] font-bold tracking-[0.1em]">CARBON_RESIDUE</span>
                      <span className="text-[10px] text-[#ffb4ab] font-bold tracking-[0.1em]">0.001%</span>
                    </div>
                    <div className="h-1 bg-white/10 w-full">
                      <div className="h-full bg-[#ffb4ab]" style={{ width: '1%' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>

        {/* BottomNavBar (Mobile Only) */}
        <nav className="fixed bottom-0 left-0 w-full flex justify-around items-center h-16 bg-[#0e0e0e]/95 backdrop-blur-md border-t border-[#3b494b] z-50 md:hidden">
          <button className="flex flex-col items-center justify-center text-[#dbfcff] bg-[#00f0ff]/10 p-2 w-full h-full">
            <span className="material-symbols-outlined">monitoring</span>
            <span className="text-[9px] mt-1 font-bold tracking-[0.1em]">VIEWPORT</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#b9cacb] p-2 w-full h-full active:bg-[#353534] transition-colors">
            <span className="material-symbols-outlined">science</span>
            <span className="text-[9px] mt-1 font-bold tracking-[0.1em]">SYNTHESIS</span>
          </button>
          <button onClick={() => navigate('/synthesis-archive')} className="flex flex-col items-center justify-center text-[#b9cacb] p-2 w-full h-full active:bg-[#353534] transition-colors">
            <span className="material-symbols-outlined">inventory_2</span>
            <span className="text-[9px] mt-1 font-bold tracking-[0.1em]">ARCHIVE</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#b9cacb] p-2 w-full h-full active:bg-[#353534] transition-colors">
            <span className="material-symbols-outlined">terminal</span>
            <span className="text-[9px] mt-1 font-bold tracking-[0.1em]">ECHO</span>
          </button>
        </nav>

        {/* UI Accents (Corner Brackets) */}
        <div className="fixed top-20 left-4 pointer-events-none opacity-20 hidden md:block">
          <div className="w-8 h-8 border-t-2 border-l-2 border-[#dbfcff]"></div>
        </div>
        <div className="fixed top-20 right-4 pointer-events-none opacity-20 hidden md:block">
          <div className="w-8 h-8 border-t-2 border-r-2 border-[#dbfcff]"></div>
        </div>
      </div>
    </>
  );
}
