import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router';
import { useLabHealth } from '../../lib/hooks';
import { Image3DViewer } from '../components/Image3DViewer';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function XRayDiffractionLab() {
  const navigate = useNavigate();
  const { health, loading } = useLabHealth('xrd');
  const [scanAngle, setScanAngle] = useState(42.84);
  const [scanProgress, setScanProgress] = useState(45);

  useEffect(() => {
    const interval = setInterval(() => {
      setScanAngle((prev) => (prev >= 90 ? 10 : prev + 0.1));
      setScanProgress((prev) => ((scanAngle - 10) / 80) * 100);
    }, 50);
    return () => clearInterval(interval);
  }, [scanAngle]);

  return (
    <div className="min-h-screen qulab-page-bg text-foreground font-['JetBrains_Mono'] selection:bg-[rgba(0,219,233,0.3)]">
      <style>{`
        .scan-line {
          background: linear-gradient(to right, transparent, #00dbe9, transparent);
          height: 1px;
          width: 100%;
          position: absolute;
          top: 50%;
          animation: scan 4s linear infinite;
        }
        @keyframes scan {
          0% { top: 0%; opacity: 0; }
          50% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        .progress-segment {
          width: 8px;
          height: 100%;
          margin-right: 2px;
        }
      `}</style>

      {/* TOP APP BAR */}
      <header className="fixed top-0 w-full z-50 bg-[rgba(19,19,19,0.8)] backdrop-blur-xl border-b border-[rgba(59,73,75,0.3)] flex justify-between items-center px-4 md:px-8 py-2">
        <div className="flex items-center gap-2 text-[#00dbe9]">
          <span className="material-symbols-outlined">terminal</span>
          <span className="font-['Space_Grotesk'] text-[20px] leading-[1.2] font-semibold tracking-tighter font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold">QULAB_INFINITE_OS</span>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden md:flex gap-6">
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#849495] hover:text-[#00dbe9] cursor-pointer transition-colors">DIAGNOSTICS</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00dbe9] border-b border-[#00dbe9]">X-RAY_DIFFRACTION</span>
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#849495] hover:text-[#00dbe9] cursor-pointer transition-colors">LATTICE_GEN</span>
          </div>
          <span className="material-symbols-outlined text-[#00dbe9]">encrypted</span>
        </div>
      </header>

            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-20 pb-24 px-4 md:px-8 max-w-7xl mx-auto space-y-2">
        {/* MAIN XRD VISUALIZER SECTION */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-2 h-auto lg:h-[600px]">
          {/* CENTRAL XRD VISUALIZER */}
          <section className="lg:col-span-8 glass-panel relative overflow-hidden group">
            {/* SAAG Header */}
            <div className="absolute top-4 left-4 z-10 flex flex-col">
              <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[rgba(0,219,233,0.8)]">SENSOR_INPUT_STREAM</span>
              <span className="font-['JetBrains_Mono'] text-[24px] leading-none tracking-[-0.05em] font-medium text-[#00dbe9]">LIVE_DIFFRACTION_MAP</span>
            </div>
            <div className="absolute top-4 right-4 z-10 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#ecffe3] shadow-[0_0_8px_#72ff70]"></span>
              <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#ecffe3]">ACTIVE_SCAN</span>
            </div>

            {/* XRD IMAGE */}
            <div className="w-full h-full flex items-center justify-center p-4">
              <div className="relative w-full h-full max-w-[500px] aspect-square">
                <Image3DViewer
                  imageUrl="https://lh3.googleusercontent.com/aida-public/AB6AXuADIjJS_2L6WVFMWrF5yvMEZfENPsSVtUNwRHNg4KJuVLbv54Bwx8ZobI4QsH0fiQklgJ0wJHB3NEqGMmt7rTWgw0h3u1SJYT6fiqRMFQ3PnZM4GYr2bqTfhqgpzPCThZAgDv9KE24o5TsgbOhRXFBwSUJw67umlh60ELQB3tX7sGP8dL33ZUrIaYfUVJMESh4L98XdW9K_LyEt_wZjsPUqGSSwf5aaLMsuV9_Ui_JOnDpyExx4rgfhX2uuEDewzV8ef7Y3oUw5kEg"
                  alt="XRD Pattern"
                  className="w-full h-full object-contain mix-blend-screen opacity-90 border border-[rgba(0,219,233,0.1)]"
                  autoRotate={true}
                />
                <div className="scan-line"></div>

                {/* PEAK MARKERS OVERLAY */}
                <div className="absolute top-1/4 left-1/4 -translate-x-1/2 -translate-y-1/2 text-[10px] text-[#00dbe9] font-['JetBrains_Mono'] border-l border-b border-[rgba(0,219,233,0.4)] pl-1 pb-1">
                  (1 0 1) INT: 0.94
                </div>
                <div className="absolute top-1/2 right-1/4 translate-x-1/2 -translate-y-1/2 text-[10px] text-[#00dbe9] font-['JetBrains_Mono'] border-r border-t border-[rgba(0,219,233,0.4)] pr-1 pt-1">
                  (2 1 1) INT: 0.72
                </div>
                <div className="absolute bottom-1/3 left-1/2 -translate-x-1/2 text-[10px] text-[#00dbe9] font-['JetBrains_Mono'] border-l border-t border-[rgba(0,219,233,0.4)] pl-1 pt-1">
                  (0 0 2) INT: 1.00
                </div>
              </div>
            </div>

            {/* 2THETA PROGRESS BAR */}
            <div className="absolute bottom-4 left-4 right-4">
              <div className="flex justify-between items-center mb-1">
                <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#849495]">2θ_SCAN_RANGE [10.00° - 90.00°]</span>
                <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00dbe9]">{scanAngle.toFixed(2)}°</span>
              </div>
              <div className="h-2 w-full flex bg-[#353534]">
                <div className="h-full bg-[#00dbe9] shadow-[0_0_10px_#00dbe9]" style={{ width: `${scanProgress}%` }}></div>
                <div className="flex-1 flex px-1">
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="progress-segment bg-[rgba(59,73,75,0.3)]"></div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* TELEMETRY SIDEBAR */}
          <aside className="lg:col-span-4 flex flex-col gap-2">
            {/* PARAMETERS TILE */}
            <div className="glass-panel p-4 flex-1">
              <div className="flex justify-between items-start mb-4">
                <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#849495]">LATTICE_PARAMETERS</span>
                <span className="material-symbols-outlined text-[#849495] text-sm">settings</span>
              </div>
              <div className="space-y-4">
                <div className="border-l-2 border-[#00dbe9] pl-3">
                  <p className="font-['JetBrains_Mono'] text-[10px] tracking-[0.1em] font-bold text-[#849495]">UNIT_CELL_A</p>
                  <p className="font-['JetBrains_Mono'] text-[20px] leading-[1.2] font-semibold text-[#00dbe9]">5.431 Å</p>
                </div>
                <div className="border-l-2 border-[#3b494b] pl-3">
                  <p className="font-['JetBrains_Mono'] text-[10px] tracking-[0.1em] font-bold text-[#849495]">CRYSTAL_SYSTEM</p>
                  <p className="font-['JetBrains_Mono'] text-[20px] leading-[1.2] font-semibold text-[#00dbe9]">CUBIC (Fd-3m)</p>
                </div>
                <div className="border-l-2 border-[#3b494b] pl-3">
                  <p className="font-['JetBrains_Mono'] text-[10px] tracking-[0.1em] font-bold text-[#849495]">PEAK_WIDTH_FWHM</p>
                  <p className="font-['JetBrains_Mono'] text-[20px] leading-[1.2] font-semibold text-[#00dbe9]">0.124°</p>
                </div>
                <div className="pt-4 space-y-2">
                  <div className="flex justify-between font-['JetBrains_Mono'] text-[10px] tracking-[0.1em] font-bold text-[#849495]">
                    <span>X-RAY_SOURCE</span>
                    <span className="text-[#00dbe9]">Cu-Kα</span>
                  </div>
                  <div className="flex justify-between font-['JetBrains_Mono'] text-[10px] tracking-[0.1em] font-bold text-[#849495]">
                    <span>VOLTAGE</span>
                    <span className="text-[#00dbe9]">45 kV</span>
                  </div>
                  <div className="flex justify-between font-['JetBrains_Mono'] text-[10px] tracking-[0.1em] font-bold text-[#849495]">
                    <span>CURRENT</span>
                    <span className="text-[#00dbe9]">40 mA</span>
                  </div>
                </div>
              </div>
            </div>

            {/* TACTICAL ACTION */}
            <button className="w-full py-6 glass-panel border-[#00dbe9] bg-[rgba(0,219,233,0.05)] hover:bg-[rgba(0,219,233,0.1)] transition-all active:scale-[0.98] group flex items-center justify-center gap-3 relative overflow-hidden">
              <div className="absolute inset-0 bg-[rgba(0,219,233,0.05)] opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <span className="material-symbols-outlined text-[#00dbe9]">refresh</span>
              <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#00dbe9] tracking-widest">RECALIBRATE_BEAM</span>
              <div className="absolute right-0 top-0 h-full w-1 bg-[#00dbe9] shadow-[0_0_15px_#00dbe9]"></div>
            </button>
          </aside>
        </div>

        {/* LOWER SECTION: REASONING LOG & CHIPS */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          {/* ECHO REASONING LOG */}
          <div className="md:col-span-2 glass-panel p-4 h-64 overflow-hidden flex flex-col">
            <div className="flex items-center gap-2 mb-2">
              <span className="material-symbols-outlined text-[#ecffe3] text-sm">psychology</span>
              <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#ecffe3]">ECHO_REASONING_ENGINE</span>
            </div>
            <div className="flex-1 font-['JetBrains_Mono'] text-xs text-[#b9cacb] overflow-y-auto space-y-2 leading-relaxed">
              <p className="text-[#00dbe9]">&gt;&gt; INITIALIZING_STRUCTURAL_ANALYSIS...</p>
              <p>&gt; Comparing current diffraction pattern against database_v4.2 (Silicon-Diamond Lattice).</p>
              <p>&gt; <span className="text-[#00dbe9]">OBSERVATION:</span> Slight shift in (1 1 1) peak noted at 28.44°. Calculating lattice strain vectors.</p>
              <p>&gt; <span className="text-[#ffb4ab]">ANOMALY DETECTED:</span> Diffuse scattering observed between peaks suggests a non-zero concentration of <span className="bg-[rgba(255,180,171,0.1)] text-[#ffb4ab] px-1">LATTICE_VACANCIES</span>.</p>
              <p>&gt; Point defect density estimated at 1.4e18 cm⁻³. This correlates with the increased thermal signature in the secondary cooling loop.</p>
              <p>&gt; Structural phase transition to metastable Beta-phase is currently suppressed but remains a 12.4% probability under current beam intensity.</p>
              <p>&gt; Recommendation: Maintain flux but prepare for recalibration if FWHM exceeds 0.135°.</p>
              <div className="w-2 h-4 bg-[#00dbe9] inline-block animate-pulse align-middle"></div>
            </div>
          </div>

          {/* STATUS CHIPS & GRID */}
          <div className="glass-panel p-4 space-y-4">
            <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold text-[#849495]">SYSTEM_STATUS</span>
            <div className="grid grid-cols-2 gap-2">
              <div className="p-2 border border-[rgba(59,73,75,0.3)] bg-[#0e0e0e] flex flex-col gap-1">
                <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#849495]">COOLING</span>
                <div className="flex items-center justify-between">
                  <span className="font-['JetBrains_Mono'] text-xs text-[#00dbe9]">OPTIMAL</span>
                  <span className="material-symbols-outlined text-[14px] text-[#ecffe3]" style={{ fontVariationSettings: "'FILL' 1" }}>check_circle</span>
                </div>
              </div>
              <div className="p-2 border border-[rgba(59,73,75,0.3)] bg-[#0e0e0e] flex flex-col gap-1">
                <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#849495]">VACUUM</span>
                <div className="flex items-center justify-between">
                  <span className="font-['JetBrains_Mono'] text-xs text-[#00dbe9]">10^-6 TORR</span>
                  <span className="material-symbols-outlined text-[14px] text-[#ecffe3]" style={{ fontVariationSettings: "'FILL' 1" }}>check_circle</span>
                </div>
              </div>
              <div className="p-2 border border-[rgba(59,73,75,0.3)] bg-[#0e0e0e] flex flex-col gap-1">
                <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#849495]">SHUTTER</span>
                <div className="flex items-center justify-between">
                  <span className="font-['JetBrains_Mono'] text-xs text-[#ffb4ab]">OPEN</span>
                  <span className="material-symbols-outlined text-[14px] text-[#ffb4ab]" style={{ fontVariationSettings: "'FILL' 1" }}>warning</span>
                </div>
              </div>
              <div className="p-2 border border-[rgba(59,73,75,0.3)] bg-[#0e0e0e] flex flex-col gap-1">
                <span className="font-['JetBrains_Mono'] text-[9px] tracking-[0.1em] font-bold text-[#849495]">GONIO</span>
                <div className="flex items-center justify-between">
                  <span className="font-['JetBrains_Mono'] text-xs text-[#00dbe9]">LOCKED</span>
                  <span className="material-symbols-outlined text-[14px] text-[#00dbe9]" style={{ fontVariationSettings: "'FILL' 1" }}>lock</span>
                </div>
              </div>
            </div>
            <div className="pt-2">
              <div className="flex items-center gap-2 px-2 py-1 bg-[rgba(0,219,233,0.1)] border border-[rgba(0,219,233,0.2)]">
                <span className="font-['JetBrains_Mono'] text-[10px] tracking-[0.1em] font-bold text-[#00dbe9]">AUTO_PHASE_ID: ENABLED</span>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* BOTTOM NAV BAR */}
      <nav className="fixed bottom-0 w-full z-50 bg-[rgba(32,31,31,0.6)] backdrop-blur-2xl border-t border-[rgba(59,73,75,0.3)] h-16 flex justify-around items-center">
        <div className="flex flex-col items-center justify-center text-[#849495] hover:text-[#7df4ff] transition-all cursor-pointer" onClick={() => navigate('/dashboard')}>
          <span className="material-symbols-outlined">grid_view</span>
          <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold mt-1">DASHBOARD</span>
        </div>
        <div className="flex flex-col items-center justify-center text-[#00dbe9] bg-[rgba(0,219,233,0.05)] rounded-none border-t-2 border-[#00dbe9] h-full px-4">
          <span className="material-symbols-outlined">biotech</span>
          <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold mt-1">LABS</span>
        </div>
        <div className="flex flex-col items-center justify-center text-[#849495] hover:text-[#7df4ff] transition-all cursor-pointer" onClick={() => navigate('/mission')}>
          <span className="material-symbols-outlined">target</span>
          <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold mt-1">MISSION</span>
        </div>
        <div className="flex flex-col items-center justify-center text-[#849495] hover:text-[#7df4ff] transition-all cursor-pointer" onClick={() => navigate('/settings')}>
          <span className="material-symbols-outlined">settings</span>
          <span className="font-['JetBrains_Mono'] text-[11px] tracking-[0.1em] font-bold mt-1">SYSTEM</span>
        </div>
      </nav>

      {/* FAB Contextual */}
      <button className="fixed bottom-20 right-6 w-14 h-14 bg-[#00dbe9] text-[#00363a] rounded-none shadow-[0_0_20px_rgba(0,219,233,0.4)] flex items-center justify-center active:scale-90 transition-transform md:hidden">
        <span className="material-symbols-outlined">bolt</span>
      </button>
    </div>
  );
}
