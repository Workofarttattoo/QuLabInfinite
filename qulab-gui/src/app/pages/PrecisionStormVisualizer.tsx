import { useLabHealth } from '../../lib/hooks';
import { Link, useNavigate } from 'react-router';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function PrecisionStormVisualizer() {
  const { health, loading } = useLabHealth('quantum');
  const navigate = useNavigate();

  return (
    <div className="font-['Inter'] text-[#dce4e5] antialiased overflow-x-hidden">
      <style>{`
        body {
          background-color: #0d1515;
          background-image: radial-gradient(#1a2426 1px, transparent 1px);
          background-size: 24px 24px;
        }
        .tech-line {
          background: rgba(0, 240, 255, 0.2);
          height: 1px;
          width: 100%;
        }
        .coordinate-marker {
          font-family: 'JetBrains Mono', monospace;
          font-size: 8px;
          color: #849495;
          position: absolute;
          opacity: 0.6;
        }
      `}</style>

      <div className="min-h-screen qulab-page-bg">
        {/* Top Navigation Shell */}
        <header className="fixed top-0 left-0 w-full z-50 bg-[#101415]/80 backdrop-blur-xl border-b border-[#00f0ff]/20">
          <div className="flex justify-between items-center px-8 h-20 max-w-[1440px] mx-auto">
            <div className="font-['Space_Grotesk'] text-2xl font-bold tracking-tighter text-[#7df4ff]">
              QULAB AIOS
            </div>
            <nav className="hidden md:flex gap-6 items-center">
              <Link to="/labs/precision-storm" className="text-[#dbfcff] border-b-2 border-[#00dbe9] pb-1 font-['JetBrains_Mono'] text-xs font-bold tracking-[0.1em] uppercase">TACTICAL</Link>
              <Link to="/labs" className="text-[#b9cacb] font-['JetBrains_Mono'] text-xs font-bold tracking-[0.1em] uppercase hover:text-[#dbfcff] transition-colors">RESEARCH</Link>
              <Link to="/echo-mission" className="text-[#b9cacb] font-['JetBrains_Mono'] text-xs font-bold tracking-[0.1em] uppercase hover:text-[#dbfcff] transition-colors">MISSION</Link>
              <Link to="/agent-telemetry" className="text-[#b9cacb] font-['JetBrains_Mono'] text-xs font-bold tracking-[0.1em] uppercase hover:text-[#dbfcff] transition-colors">TELEMETRY</Link>
            </nav>
            <div className="flex items-center gap-4">
              <button
                onClick={() => navigate('/labs')}
                className="bg-[#00f0ff] text-[#004f54] px-6 py-2.5 font-['JetBrains_Mono'] text-xs font-bold tracking-[0.1em] uppercase hover:brightness-110 transition-all active:scale-95"
              >
                RETURN TO LABS
              </button>
            </div>
          </div>
        </header>

              <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-20">
          {/* Hero Section */}
          <section className="relative min-h-[921px] flex items-center justify-center px-8 overflow-hidden">
            <div className="max-w-[1200px] w-full grid grid-cols-1 lg:grid-cols-2 gap-6 items-center relative z-10">
              <div className="space-y-6">
                <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#dbfcff]/10 border border-[#dbfcff]/20 text-[#00dbe9] font-['JetBrains_Mono'] text-xs font-medium rounded-full">
                  <span className="w-2 h-2 rounded-full bg-[#dbfcff] animate-pulse"></span>
                  SYSTEM_OPERATIONAL_01
                </div>
                <h1 className="font-['Space_Grotesk'] text-5xl font-bold text-[#dbfcff] leading-none">
                  Outpace the Unknown.<br/>Dominate the Lab.
                </h1>
                <p className="font-['Inter'] text-lg text-[#b9cacb] max-w-lg">
                  Accelerate quantum research through unified tactical intelligence. QULAB AIOS harmonizes agent fleets and qubit stability protocols in a single dominance-first OS.
                </p>
                <div className="flex gap-4 pt-4">
                  <button className="bg-[#00f0ff] text-[#004f54] px-8 py-4 font-['JetBrains_Mono'] text-xs font-bold tracking-widest hover:brightness-110">INITIATE VOID PROTOCOL</button>
                  <button className="border border-[#dbfcff]/40 bg-transparent px-8 py-4 font-['JetBrains_Mono'] text-xs text-[#dbfcff] hover:bg-[#dbfcff]/10 backdrop-blur-md">VIEW ARSENAL</button>
                </div>
              </div>
              <div className="relative group">
                <div className="glass-panel p-8 rounded-lg relative aspect-square overflow-hidden flex items-center justify-center">
                  <span className="coordinate-marker top-2 left-2">LAT: 34.0522° N</span>
                  <span className="coordinate-marker bottom-2 right-2">LONG: 118.2437° W</span>
                  <img
                    alt="Neural Mesh Infographic"
                    className="w-full h-full object-cover opacity-80 mix-blend-screen"
                    src="https://lh3.googleusercontent.com/aida-public/AB6AXuCVjxUhOeexB9vjqqAI-D1roTHJ1Dh6HWKhPNrke6ZDUng-IZA0xoK5W2d3rwVYmz2NFVv7k0xeD3PFvMQoSiAaRqP739zbiJ7uuRAxHwqssSxCK9tNl2rdZyLNJeNRtSoFKvlAR7U8Ka6nD5QN-nTee7T5sKJsvlHhQLRZr7KP4c0irlKZugWw3DhDtOTYmdCPoiDcvQtfZ82UlODbdOdbmSPTRsQzw5SmLR-eTPtKAPDbPFcTIRYdOo81rQiAvAtPIgLWD2um7iE"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-[#101415] via-transparent to-transparent"></div>
                  <div className="absolute inset-0 border border-[#dbfcff]/20 pointer-events-none"></div>
                </div>
              </div>
            </div>
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[140%] h-[140%] opacity-20 pointer-events-none">
              <div className="w-full h-full border-[1px] border-[#dbfcff]/5 rounded-full animate-spin" style={{ animation: 'spin 20s linear infinite' }}></div>
            </div>
          </section>

          {/* Live Tactical Intelligence */}
          <section className="py-24 px-8 bg-[#080f10]">
            <div className="max-w-[1440px] mx-auto">
              <div className="mb-12 flex items-end justify-between border-b border-[#3b494b]/30 pb-4">
                <div>
                  <p className="font-['JetBrains_Mono'] text-[#dbfcff] text-sm mb-2">// STREAM_FEED: ACTIVE</p>
                  <h2 className="font-['Space_Grotesk'] text-3xl font-semibold">Live Tactical Intelligence</h2>
                </div>
                <div className="font-['JetBrains_Mono'] text-[#849495] text-xs text-right">
                  REFRESH_RATE: 0.003ms<br/>
                  ENCRYPTION: AES-XTS-512
                </div>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <div className="lg:col-span-3 glass-panel p-1 rounded-lg relative min-h-[600px] flex flex-col">
                  <span className="coordinate-marker top-2 left-2">SYS_LOCK: 14.22.0</span>
                  <span className="coordinate-marker bottom-2 right-2">GRID_REF: AX-99</span>
                  <div className="flex-1 bg-black/40 p-8 flex items-center justify-center">
                    <div className="text-center space-y-4">
                      <span className="material-symbols-outlined text-[#dbfcff] text-6xl animate-pulse">monitoring</span>
                      <p className="font-['JetBrains_Mono'] text-[#849495] text-sm tracking-widest">
                        {health ? `PROCESSING QUANTUM DATA... ${health.qubitStability || '98.42%'} STABILITY` : 'AWAITING EXTERNAL TELEMETRY LINK...'}
                      </p>
                    </div>
                  </div>
                </div>
                <div className="space-y-6">
                  <div className="glass-panel p-6 rounded-lg relative">
                    <h4 className="font-['JetBrains_Mono'] text-xs text-[#dbfcff] mb-4 border-b border-[#dbfcff]/20 pb-2">THREAT_VECTOR_ANALYSIS</h4>
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="font-['JetBrains_Mono'] text-[11px] text-[#b9cacb]">NODE_ALPHA</span>
                        <span className="font-['JetBrains_Mono'] text-[11px] text-[#dbfcff]">0.04% VAR</span>
                      </div>
                      <div className="w-full bg-[#3b494b]/30 h-1">
                        <div className="bg-[#dbfcff] h-full w-[45%]"></div>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="font-['JetBrains_Mono'] text-[11px] text-[#b9cacb]">NODE_SIGMA</span>
                        <span className="font-['JetBrains_Mono'] text-[11px] text-[#dbfcff]">0.98% VAR</span>
                      </div>
                      <div className="w-full bg-[#3b494b]/30 h-1">
                        <div className="bg-[#dbfcff] h-full w-[82%]"></div>
                      </div>
                    </div>
                  </div>
                  <div className="glass-panel p-6 rounded-lg relative overflow-hidden">
                    <div className="absolute -right-8 -bottom-8 opacity-10">
                      <span className="material-symbols-outlined text-[120px]">security</span>
                    </div>
                    <h4 className="font-['JetBrains_Mono'] text-xs text-[#dbfcff] mb-4 border-b border-[#dbfcff]/20 pb-2">ENCRYPTION_STATUS</h4>
                    <p className="font-['JetBrains_Mono'] text-xs text-[#b9cacb] leading-relaxed">
                      QUANTUM KEY DISTRIBUTION: <span className="text-[#dbfcff]">SECURED</span><br/>
                      PHOTON ENTANGLEMENT: 99.9%<br/>
                      INTERCEPT_ATTEMPTS: 0
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* The Research Arsenal */}
          <section className="py-24 px-8 bg-[#0d1515]">
            <div className="max-w-[1440px] mx-auto text-center mb-16">
              <h2 className="font-['Space_Grotesk'] text-3xl font-semibold text-[#dbfcff] mb-4">The Research Arsenal</h2>
              <p className="text-[#b9cacb] max-w-2xl mx-auto">High-fidelity visualization components for real-time monitoring of quantum states and agent deployment.</p>
            </div>
            <div className="max-w-[1440px] mx-auto grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="glass-panel p-8 rounded-lg relative flex flex-col group hover:border-[#dbfcff]/50 transition-all duration-300">
                <span className="coordinate-marker top-2 left-2">INDEX: QB-S1</span>
                <h3 className="font-['Space_Grotesk'] text-2xl mb-6 text-[#dbfcff]">Qubit Stability Index</h3>
                <div className="flex-1 bg-[#232b2c] rounded-sm p-4 relative overflow-hidden border border-[#3b494b]/20">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <svg className="w-full h-full" viewBox="0 0 200 100">
                      <path className="drop-shadow-[0_0_8px_rgba(0,240,255,0.5)]" d="M0,80 Q25,30 50,70 T100,20 T150,50 T200,10" fill="none" stroke="#00f0ff" strokeWidth="1"></path>
                      <path d="M0,85 Q25,35 50,75 T100,25 T150,55 T200,15" fill="none" opacity="0.3" stroke="#00f0ff" strokeWidth="1"></path>
                    </svg>
                  </div>
                  <div className="absolute top-4 right-4 flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-[#dbfcff] animate-ping"></div>
                    <span className="font-['JetBrains_Mono'] text-[10px] text-[#dbfcff]">LIVE_TRACK</span>
                  </div>
                </div>
                <div className="mt-6 flex justify-between font-['JetBrains_Mono'] text-xs">
                  <span className="text-[#b9cacb]">STABILITY_MEAN</span>
                  <span className="text-[#dbfcff] font-bold">{health?.qubitStability || '98.42%'}</span>
                </div>
              </div>

              <div className="glass-panel p-8 rounded-lg relative flex flex-col group hover:border-[#dbfcff]/50 transition-all duration-300">
                <span className="coordinate-marker top-2 left-2">INDEX: MAT-HM</span>
                <h3 className="font-['Space_Grotesk'] text-2xl mb-6 text-[#dbfcff]">Success Heatmap</h3>
                <div className="grid grid-cols-6 gap-2">
                  {[40, 20, 60, 10, 80, 30, 10, 90, 20, 40, 10, 20].map((opacity, i) => (
                    <div key={i} className="aspect-square rounded-sm" style={{ background: `rgba(0, 240, 255, ${opacity / 100})` }}></div>
                  ))}
                </div>
                <p className="mt-6 font-['JetBrains_Mono'] text-xs text-[#b9cacb]">LATTICE_INTEGRITY: OPTIMAL</p>
              </div>

              <div className="glass-panel p-8 rounded-lg relative flex flex-col group hover:border-[#dbfcff]/50 transition-all duration-300">
                <span className="coordinate-marker top-2 left-2">INDEX: AF-U</span>
                <h3 className="font-['Space_Grotesk'] text-2xl mb-6 text-[#dbfcff]">Fleet Utilization</h3>
                <div className="flex-1 flex items-center justify-center">
                  <div className="relative w-40 h-40">
                    <div className="absolute inset-0 border-4 border-[#dbfcff]/20 rounded-full"></div>
                    <div className="absolute inset-1 border-4 border-[#dbfcff] border-t-transparent rounded-full rotate-45"></div>
                    <div className="absolute inset-4 border-4 border-[#3b494b]/30 rounded-full"></div>
                    <div className="absolute inset-5 border-4 border-[#00dbe9] border-b-transparent rounded-full -rotate-90"></div>
                    <div className="absolute inset-0 flex items-center justify-center flex-col">
                      <span className="font-['Space_Grotesk'] text-2xl text-[#dbfcff]">87%</span>
                      <span className="font-['JetBrains_Mono'] text-[9px] text-[#849495]">CAPACITY</span>
                    </div>
                  </div>
                </div>
                <p className="mt-6 font-['JetBrains_Mono'] text-xs text-[#b9cacb] text-center">342 ACTIVE AGENTS ONLINE</p>
              </div>
            </div>
          </section>

          {/* Footer */}
          <footer className="bg-[#0d1515] border-t border-[#3b494b]/20 w-full py-12 flex flex-col items-center gap-6 px-8">
            <div className="text-[#7df4ff] font-['Space_Grotesk'] text-2xl">QULAB AIOS</div>
            <div className="flex gap-8">
              <Link to="/labs" className="font-['JetBrains_Mono'] text-xs text-[#b9cacb] hover:text-[#7df4ff] transition-all opacity-80 hover:opacity-100">PROTOCOLS</Link>
              <Link to="/system-lockdown" className="font-['JetBrains_Mono'] text-xs text-[#b9cacb] hover:text-[#7df4ff] transition-all opacity-80 hover:opacity-100">ENCRYPTION</Link>
              <Link to="/synthesis-archive" className="font-['JetBrains_Mono'] text-xs text-[#b9cacb] hover:text-[#7df4ff] transition-all opacity-80 hover:opacity-100">MISSION_LOGS</Link>
            </div>
            <div className="font-['JetBrains_Mono'] text-xs text-[#849495] text-center opacity-60">
              © 2024 QULAB AIOS // TACTICAL DOMINANCE UNIT<br/>
              <span className="text-[10px]">ALL SYSTEMS MONITORED // SECURE ENCLAVE ACTIVE</span>
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}
