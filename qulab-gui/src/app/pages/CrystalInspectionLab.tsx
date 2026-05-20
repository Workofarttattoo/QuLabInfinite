import { Link, useNavigate } from 'react-router';
import { Image3DViewer } from '../components/Image3DViewer';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function CrystalInspectionLab() {
  const navigate = useNavigate();
  const labHealth = useLabHealth('chemistry');

  return (
    <div className="min-h-screen qulab-page-bg text-foreground font-['JetBrains_Mono'] overflow-hidden">
      <style>{`
        .scanline {
          background: linear-gradient(to bottom, transparent 50%, rgba(0, 240, 255, 0.05) 50%);
          background-size: 100% 4px;
        }
        .segmented-bar div {
          width: 8px;
          height: 100%;
          background: #00f0ff;
          opacity: 0.3;
          margin-right: 2px;
        }
        .segmented-bar div.active {
          opacity: 1;
          box-shadow: 0 0 8px #00f0ff;
        }
        .status-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
        }
        .cursor-blink {
          display: inline-block;
          width: 1ch;
          height: 1.2em;
          background: #00f0ff;
          animation: blink 1s step-end infinite;
        }
        @keyframes blink { 50% { opacity: 0; } }
      `}</style>

      {/* TopAppBar */}
      <header className="fixed top-0 w-full z-50 bg-[#0d1515]/80 backdrop-blur-xl border-b border-[#3b494b]/30 flex justify-between items-center px-4 md:px-8 h-16">
        <div className="flex items-center gap-4">
          <span className="material-symbols-outlined text-[#dbfcff]" style={{ fontVariationSettings: "'FILL' 1" }}>biotech</span>
          <h1 className="font-['Space_Grotesk'] text-3xl font-semibold tracking-tighter text-[#dbfcff] uppercase">QULAB_INFINITE_OS</h1>
        </div>
        <div className="hidden md:flex gap-8">
          <button className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#dbfcff] border-b-2 border-[#dbfcff] py-1 uppercase">Telemetry</button>
          <button className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#b9cacb] hover:text-[#dbfcff] transition-colors py-1 uppercase">Synthesis</button>
          <button className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#b9cacb] hover:text-[#dbfcff] transition-colors py-1 uppercase">Protocol</button>
          <button className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#b9cacb] hover:text-[#dbfcff] transition-colors py-1 uppercase">Archive</button>
        </div>
        <div className="flex items-center gap-4">
          <span className="material-symbols-outlined text-[#dbfcff] cursor-pointer hover:scale-110 duration-150">settings_input_component</span>
        </div>
      </header>

      {/* Side Navigation */}
      <aside className="fixed left-0 top-16 h-[calc(100vh-64px)] w-64 z-40 bg-[#1c1b1b]/90 backdrop-blur-lg border-r border-[#3b494b]/20 hidden md:flex flex-col p-2">
        <div className="mb-6 px-2">
          <h2 className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#b9cacb]/60 uppercase mb-2">System Core</h2>
          <div className="font-['Space_Grotesk'] text-xl font-semibold text-[#dbfcff]">SYSTEM_CORE_V1</div>
        </div>
        <nav className="flex flex-col gap-1">
          <button
            onClick={() => navigate('/labs')}
            className="flex items-center gap-3 p-3 bg-[#13ff43]/20 text-[#007117] border-l-2 border-[#00e639] transition-all duration-200"
          >
            <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>analytics</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold uppercase">Telemetry</span>
          </button>
          <Link to="/labs/universal-chemistry" className="flex items-center gap-3 p-3 text-[#b9cacb] hover:bg-[#353534]/40 transition-all duration-200">
            <span className="material-symbols-outlined">science</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold uppercase">Synthesis</span>
          </Link>
          <Link to="/labs/crystal-inspection" className="flex items-center gap-3 p-3 text-[#b9cacb] hover:bg-[#353534]/40 transition-all duration-200">
            <span className="material-symbols-outlined">query_stats</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold uppercase">Stability</span>
          </Link>
          <Link to="/labs/xrd" className="flex items-center gap-3 p-3 text-[#b9cacb] hover:bg-[#353534]/40 transition-all duration-200">
            <span className="material-symbols-outlined">list_alt</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold uppercase">Protocol</span>
          </Link>
          <Link to="/synthesis-archive" className="flex items-center gap-3 p-3 text-[#b9cacb] hover:bg-[#353534]/40 transition-all duration-200">
            <span className="material-symbols-outlined">history</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold uppercase">Archive</span>
          </Link>
        </nav>
      </aside>

      {/* Main Canvas */}
            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-16 md:pl-64 h-screen w-full relative overflow-hidden flex flex-col">
        <div className="flex-1 grid grid-cols-12 grid-rows-6 p-4 gap-4">
          {/* Holographic Viewport (The Centerpiece) */}
          <section className="col-span-12 md:col-span-8 row-span-4 relative overflow-hidden glass-panel group">
            <div className="absolute top-2 left-2 z-10 flex items-center gap-2">
              <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#dbfcff] uppercase">Viewport: ATOMIC_STRUCTURE_01</span>
            </div>
            <div className="absolute top-2 right-2 z-10">
              <div className="flex items-center gap-2 bg-black/40 px-2 py-1">
                <div className="status-dot bg-[#00e639] shadow-[0_0_5px_#00e639]"></div>
                <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00e639]">LIVE_STREAMING</span>
              </div>
            </div>

            {/* Overlay Markers */}
            <div className="absolute inset-0 z-20 pointer-events-none p-8 flex flex-col justify-between">
              <div className="flex justify-between items-start">
                <div className="border-l border-t border-[#dbfcff]/40 w-16 h-16 p-2">
                  <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#dbfcff]/60">X: 104.22</span>
                </div>
                <div className="border-r border-t border-[#dbfcff]/40 w-16 h-16 p-2 text-right">
                  <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#dbfcff]/60">Y: 882.10</span>
                </div>
              </div>
              {/* HUD Overlays */}
              <div className="flex flex-col gap-4 self-center w-full max-w-md">
                <div className="h-[0.5px] bg-[#dbfcff]/20 w-full relative">
                  <div className="absolute -top-1 left-1/4 w-2 h-2 border border-[#dbfcff] bg-[#050505]"></div>
                  <div className="absolute -top-3 left-1/4 text-[8px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold text-[#dbfcff]">CROSS_SECTION_SCAN</div>
                </div>
              </div>
              <div className="flex justify-between items-end">
                <div className="border-l border-b border-[#dbfcff]/40 w-16 h-16 p-2 flex flex-col justify-end">
                  <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#dbfcff]/60">Z: 0.0039</span>
                </div>
                <div className="border-r border-b border-[#dbfcff]/40 w-16 h-16 p-2 text-right flex flex-col justify-end">
                  <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#dbfcff]/60">SCALE: 1:1,000,000</span>
                </div>
              </div>
            </div>

            {/* Main Image */}
            <div className="w-full h-full relative group cursor-crosshair">
              <Image3DViewer
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuBKIZqc_prrC_N9CM3EUne2wvc_CQyyv93U-dqT-mBpygnKXvYDPqsDE5zSqyRHTRNGjF7CoU6lfrsaL7hzlVnaQApwZqYiPoYo97mTHn8zIg5PEB2w4cU9Hg6tSzSzT_EhzbN8BLlG6TXo_RvyiNQBv4iO2q92y9yR2QtBBrbhdBT5iTeKhU-5ZSpZ00tQzd8NbRrMuFycjvyCCuqB1iO4C_ni12CFVwADn17tWujFJn8g0yUaAd6DQeLctKkdZioHjuOK5RdPdnQ"
                alt="Holographic crystal inspection"
                className="w-full h-full object-cover mix-blend-screen opacity-90 transition-transform duration-700 group-hover:scale-105"
                autoRotate={true}
              />
              <div className="absolute inset-0 scanline opacity-30"></div>
            </div>

            {/* Floating Readout */}
            <div className="absolute bottom-8 right-8 z-30 glass-panel p-4 w-48 border-l-2 border-l-[#dbfcff]">
              <h4 className="font-['JetBrains_Mono'] text-[10px] tracking-[0.1em] font-bold text-[#dbfcff] mb-2">BOND INTEGRITY</h4>
              <div className="space-y-2">
                <div className="flex justify-between text-[10px] font-['JetBrains_Mono'] tracking-[-0.05em] font-medium">
                  <span>COVALENT</span>
                  <span className="text-[#00f0ff]">98.4%</span>
                </div>
                <div className="h-1 bg-white/10 w-full">
                  <div className="h-full bg-[#00f0ff] w-[98%]" style={{ boxShadow: '0 0 10px #00f0ff' }}></div>
                </div>
                <div className="flex justify-between text-[10px] font-['JetBrains_Mono'] tracking-[-0.05em] font-medium">
                  <span>VAN DER WAALS</span>
                  <span className="text-[#ffb4ab]">1.2%</span>
                </div>
                <div className="h-1 bg-white/10 w-full">
                  <div className="h-full bg-[#ffb4ab] w-[12%]"></div>
                </div>
              </div>
            </div>
          </section>

          {/* ECHO_REASONING_LOG */}
          <section className="col-span-12 md:col-span-4 row-span-3 glass-panel p-4 flex flex-col">
            <div className="flex justify-between items-center mb-4 border-b border-[#3b494b]/30 pb-2">
              <h3 className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00f0ff] uppercase">ECHO_REASONING_LOG</h3>
              <span className="material-symbols-outlined text-[#00f0ff]/40 text-sm">terminal</span>
            </div>
            <div className="flex-1 overflow-y-auto space-y-4 font-['JetBrains_Mono'] text-[12px] text-[#b9cacb]/80">
              <div className="flex gap-2">
                <span className="text-[#00f0ff] shrink-0">[14:02:44]</span>
                <p>Initializing lattice deep-scan sequence. Calibration aligned to Superconducting Channel 7.</p>
              </div>
              <div className="flex gap-2 bg-[#dbfcff]/5 p-2 border-l border-[#dbfcff]">
                <span className="text-[#00f0ff] shrink-0">[14:02:51]</span>
                <p>Detected lattice vacancy at coord [45.2, 12.8]. Atomic density dropping below threshold &lt;0.84.</p>
              </div>
              <div className="flex gap-2">
                <span className="text-[#00f0ff] shrink-0">[14:03:05]</span>
                <p>Vacancy mitigation logic active. Suggesting photon-pulse realignment to stabilize the superconducting channel.</p>
              </div>
              <div className="flex gap-2">
                <span className="text-[#00f0ff] shrink-0">[14:03:12]</span>
                <p>Awaiting operator override for realignment pulse. Estimated success probability: 0.941.</p>
              </div>
              <div className="flex gap-2">
                <span className="text-[#00f0ff] shrink-0">[14:03:22]</span>
                <p>Internal coherence monitoring stable but fragile. Thermal variance minimal. <span className="cursor-blink">_</span></p>
              </div>
            </div>
          </section>

          {/* ATOMIC_STABILITY */}
          <section className="col-span-12 md:col-span-4 row-span-1 glass-panel p-4">
            <h3 className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00f0ff] uppercase mb-4">ATOMIC_STABILITY</h3>
            <div className="flex items-end h-16 gap-1">
              {/* Fake Bar Chart */}
              <div className="w-full bg-[#dbfcff]/20 h-8 relative">
                <div className="absolute bottom-0 left-0 w-2 h-12 bg-[#00f0ff] shadow-[0_0_8px_#00f0ff]"></div>
                <div className="absolute bottom-0 left-4 w-2 h-8 bg-[#00f0ff]/60"></div>
                <div className="absolute bottom-0 left-8 w-2 h-10 bg-[#00f0ff]/80"></div>
                <div className="absolute bottom-0 left-12 w-2 h-4 bg-[#ffb4ab]"></div>
                <div className="absolute bottom-0 left-16 w-2 h-14 bg-[#00f0ff] shadow-[0_0_8px_#00f0ff]"></div>
                <div className="absolute bottom-0 left-20 w-2 h-6 bg-[#00f0ff]/40"></div>
                <div className="absolute bottom-0 left-24 w-2 h-9 bg-[#00f0ff]/70"></div>
                <div className="absolute bottom-0 left-28 w-2 h-2 bg-[#ffb4ab]"></div>
                <div className="absolute bottom-0 left-32 w-2 h-11 bg-[#00f0ff]"></div>
                <div className="absolute bottom-0 left-36 w-2 h-14 bg-[#00f0ff] shadow-[0_0_8px_#00f0ff]"></div>
              </div>
              <div className="shrink-0 flex flex-col justify-end ml-4">
                <span className="font-['JetBrains_Mono'] text-2xl tracking-[-0.05em] font-medium text-[#dbfcff]">0.92</span>
                <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#dbfcff]/40">COHERENCE_SIGMA</span>
              </div>
            </div>
          </section>

          {/* TACTICAL_CONTROLS */}
          <section className="col-span-12 md:col-span-8 row-span-2 glass-panel p-4 flex flex-col justify-between">
            <div className="flex justify-between items-center mb-2">
              <h3 className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00f0ff] uppercase">TACTICAL_CONTROLS</h3>
              <div className="flex gap-2">
                <span className="px-2 py-0.5 bg-[#13ff43]/20 text-[#007117] text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold">AUTO_SCAN</span>
                <span className="px-2 py-0.5 bg-[#dbfcff]/10 text-[#dbfcff] text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold">LOCKED</span>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4 h-full pt-4">
              <button className="border border-[#00f0ff] group hover:bg-[#00f0ff]/10 transition-all flex flex-col items-center justify-center gap-2 active:scale-95 duration-150 relative">
                <div className="absolute top-1 left-1 w-1 h-1 bg-[#00f0ff]"></div>
                <span className="material-symbols-outlined text-[#dbfcff] group-hover:scale-110 duration-200">zoom_in</span>
                <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00f0ff]">ZOOM_INTERNAL</span>
              </button>
              <button className="border border-[#00f0ff] group hover:bg-[#00f0ff]/10 transition-all flex flex-col items-center justify-center gap-2 active:scale-95 duration-150 relative">
                <div className="absolute top-1 left-1 w-1 h-1 bg-[#00f0ff]"></div>
                <span className="material-symbols-outlined text-[#dbfcff] group-hover:scale-110 duration-200">3d_rotation</span>
                <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00f0ff]">ROTATE_AXIS</span>
              </button>
              <button className="border border-[#00f0ff] bg-[#00f0ff]/5 group hover:bg-[#00f0ff]/20 transition-all flex flex-col items-center justify-center gap-2 active:scale-95 duration-150 relative">
                <div className="absolute top-1 left-1 w-1 h-1 bg-[#00f0ff]"></div>
                <span className="material-symbols-outlined text-[#dbfcff] group-hover:scale-110 duration-200">visibility</span>
                <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00f0ff]">X-RAY_SCAN</span>
                <div className="absolute bottom-2 w-1/2 h-0.5 bg-[#00f0ff]/40"></div>
              </button>
            </div>
          </section>

          {/* SYNTHESIS_STEPS */}
          <section className="col-span-12 md:col-span-4 row-span-2 glass-panel p-4 flex flex-col">
            <h3 className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00f0ff] uppercase mb-4">SYNTHESIS_STEPS</h3>
            <div className="space-y-3">
              <div className="flex items-center gap-4">
                <div className="segmented-bar flex h-3 shrink-0">
                  <div className="active"></div><div className="active"></div><div className="active"></div><div className="active"></div>
                </div>
                <div className="flex-1 flex justify-between items-center border-b border-[#3b494b]/20 pb-1">
                  <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#e5e2e1]">Crystal Inoculation</span>
                  <span className="material-symbols-outlined text-[#00e639] text-sm" style={{ fontVariationSettings: "'FILL' 1" }}>check_circle</span>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="segmented-bar flex h-3 shrink-0">
                  <div className="active"></div><div className="active"></div><div></div><div></div>
                </div>
                <div className="flex-1 flex justify-between items-center border-b border-[#3b494b]/20 pb-1">
                  <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00f0ff]">Lattice Realignment</span>
                  <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#dbfcff] animate-pulse">PROCESSING</span>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="segmented-bar flex h-3 shrink-0">
                  <div></div><div></div><div></div><div></div>
                </div>
                <div className="flex-1 flex justify-between items-center border-b border-[#3b494b]/20 pb-1">
                  <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#b9cacb]">Superconducting Verify</span>
                  <span className="material-symbols-outlined text-[#b9cacb]/40 text-sm">lock</span>
                </div>
              </div>
            </div>
            <div className="mt-auto pt-4 flex gap-4">
              <div className="flex-1 border border-[#dbfcff] p-2 flex flex-col justify-center">
                <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#dbfcff]/60 uppercase">Reactor Temp</span>
                <span className="font-['JetBrains_Mono'] text-[16px] tracking-[-0.05em] font-medium text-[#dbfcff]">273.15 K</span>
              </div>
              <div className="flex-1 border border-[#3b494b]/30 p-2 flex flex-col justify-center">
                <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#b9cacb]/60 uppercase">Phase Angle</span>
                <span className="font-['JetBrains_Mono'] text-[16px] tracking-[-0.05em] font-medium text-[#e5e2e1]">12.08°</span>
              </div>
            </div>
          </section>
        </div>

        {/* Bottom Status Bar (Mobile Navigation Placeholder) */}
        <footer className="fixed bottom-0 left-0 w-full flex justify-around items-center h-16 px-4 md:hidden bg-[#0e0e0e]/90 backdrop-blur-md border-t border-[#00f0ff]/20 z-50">
          <button
            onClick={() => navigate('/labs')}
            className="flex flex-col items-center justify-center text-[#00f0ff] drop-shadow-[0_0_8px_rgba(0,240,255,0.5)] active:scale-110 duration-300"
          >
            <span className="material-symbols-outlined">view_in_ar</span>
            <span className="text-[8px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold uppercase">View</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#b9cacb]/50 hover:text-[#7df4ff] active:scale-110 duration-300">
            <span className="material-symbols-outlined">radar</span>
            <span className="text-[8px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold uppercase">Scan</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#b9cacb]/50 hover:text-[#7df4ff] active:scale-110 duration-300">
            <span className="material-symbols-outlined">terminal</span>
            <span className="text-[8px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold uppercase">Log</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#b9cacb]/50 hover:text-[#7df4ff] active:scale-110 duration-300">
            <span className="material-symbols-outlined">memory</span>
            <span className="text-[8px] font-['JetBrains_Mono'] tracking-[0.1em] font-bold uppercase">Core</span>
          </button>
        </footer>
      </main>
    </div>
  );
}
