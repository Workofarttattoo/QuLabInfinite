import { useLabHealth } from '../../lib/hooks';
import { Link, useNavigate } from 'react-router';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function TacticalIntelligenceVisualizer() {
  const { health, loading } = useLabHealth('quantum');
  const navigate = useNavigate();

  return (
    <div className="font-['Inter'] text-[#dce4e5] selection:bg-[#00f0ff] selection:text-[#00363a]">
      <style>{`
        body {
          background-color: #0a0e14;
          background-image: radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.05) 1px, transparent 0);
          background-size: 24px 24px;
        }
        .tactical-border {
          position: relative;
        }
        .tactical-border::before, .tactical-border::after {
          content: '';
          position: absolute;
          width: 10px;
          height: 10px;
          border-color: #00f0ff;
          border-style: solid;
        }
        .tactical-border::before { top: -1px; left: -1px; border-width: 2px 0 0 2px; }
        .tactical-border::after { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }

        .neon-glow-cyan {
          box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
          border: 1px solid rgba(0, 240, 255, 0.3);
        }
        .neon-glow-purple {
          box-shadow: 0 0 15px rgba(168, 85, 247, 0.1);
          border: 1px solid rgba(168, 85, 247, 0.3);
        }
        .hex-grid {
          clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);
        }
      `}</style>

      <div className="min-h-screen" style={{ backgroundColor: '#0a0e14', backgroundImage: 'radial-gradient(circle at 1px 1px, rgba(255, 255, 255, 0.05) 1px, transparent 0)', backgroundSize: '24px 24px' }}>
        {/* Top Navigation Shell */}
        <header className="fixed top-0 left-0 w-full z-50 px-12 py-4">
          <nav className="glass-panel rounded-lg flex justify-between items-center px-6 py-3 border-b border-white/10 shadow-[0_4px_30px_rgba(0,0,0,0.1)]">
            <div className="flex items-center gap-4">
              <span className="font-['Space_Grotesk'] text-3xl font-bold text-[#00dbe9] tracking-tighter">QuLab AIOS</span>
              <div className="h-4 w-[1px] bg-[#3b494b]/30 hidden md:block"></div>
              <div className="hidden md:flex gap-6">
                <Link to="/" className="font-['Space_Grotesk'] text-xs tracking-widest text-[#00dbe9] border-b-2 border-[#00dbe9] pb-1 font-bold">MISSION</Link>
                <Link to="/echo-mission" className="font-['Space_Grotesk'] text-xs tracking-widest text-[#b9cacb] hover:text-[#dbfcff] transition-all duration-200 font-bold">ORCHESTRATION</Link>
                <Link to="/intel-dashboard" className="font-['Space_Grotesk'] text-xs tracking-widest text-[#b9cacb] hover:text-[#dbfcff] transition-all duration-200 font-bold">INTELLIGENCE</Link>
                <Link to="/system-lockdown" className="font-['Space_Grotesk'] text-xs tracking-widest text-[#b9cacb] hover:text-[#dbfcff] transition-all duration-200 font-bold">INFRASTRUCTURE</Link>
              </div>
            </div>
            <div className="flex items-center gap-6">
              <div className="flex gap-4 text-[#b9cacb]">
                <span className="material-symbols-outlined hover:text-[#00dbe9] cursor-pointer transition-colors">sensors</span>
                <span className="material-symbols-outlined hover:text-[#00dbe9] cursor-pointer transition-colors">settings_input_component</span>
                <span className="material-symbols-outlined hover:text-[#00dbe9] cursor-pointer transition-colors">terminal</span>
              </div>
              <button
                onClick={() => navigate('/labs')}
                className="bg-[#00f0ff] text-[#00363a] font-['Space_Grotesk'] text-xs px-6 py-2 rounded-sm hover:opacity-90 active:scale-95 transition-all font-bold"
              >
                RETURN TO LABS
              </button>
            </div>
          </nav>
        </header>

              <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="relative pt-32 pb-20 overflow-hidden">
          {/* Hero Section: Orchestration Engine */}
          <section className="px-12 relative mb-32">
            <div className="grid grid-cols-12 gap-6 items-center">
              <div className="col-span-5 z-10">
                <div className="mb-4 inline-flex items-center gap-2 px-3 py-1 bg-[#6f00be]/20 border border-[#6f00be]/40 rounded-full">
                  <span className="w-2 h-2 rounded-full bg-[#ddb7ff] animate-pulse"></span>
                  <span className="font-['Space_Grotesk'] text-xs text-[#ddb7ff] font-bold tracking-wider">OS_VERSION_4.0_STABLE</span>
                </div>
                <h1 className="font-['Space_Grotesk'] text-5xl font-bold text-[#dbfcff] mb-6 leading-tight">
                  Autonomous Multi-Agent <br/>
                  <span className="text-[#00dbe9]">Orchestration Shell</span>
                </h1>
                <p className="font-['Inter'] text-lg text-[#b9cacb] mb-8 max-w-xl">
                  Deploy, sync, and govern swarm intelligence across distributed quantum nodes. QuLab AIOS provides the tactical substrate for hyper-scale cognitive workflows.
                </p>
                <div className="flex gap-4">
                  <button className="px-8 py-4 bg-[#00dbe9] text-[#0d1515] font-['Space_Grotesk'] text-xs font-bold hover:brightness-110 transition-all tactical-border">
                    ESTABLISH UPLINK
                  </button>
                  <button className="px-8 py-4 border border-[#849495] text-[#dce4e5] font-['Space_Grotesk'] text-xs hover:bg-white/5 transition-all">
                    VIEW PROTOCOLS
                  </button>
                </div>
              </div>
              <div className="col-span-7 relative h-[600px]">
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="relative w-full h-full glass-panel neon-glow-cyan border-white/5 rounded-xl overflow-hidden">
                    <div className="absolute top-4 left-4 font-['Space_Grotesk'] text-[10px] text-[#dbfcff]/40 uppercase tracking-tighter">
                      [ X: 142.09 // Y: 882.11 ]<br/>ORCH_MATRIX_ACTIVE
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <img
                        alt="Orchestration Hub"
                        className="w-full h-full object-cover opacity-60 mix-blend-screen"
                        src="https://lh3.googleusercontent.com/aida-public/AB6AXuAPx7KDng1WMn4mTe00YezYh3HJrzAuba-D0IhED3DlrPo0Y0x81eQsgadPfDT4WJJiMphwAJv80jgXyqc4rL3KP4-irPUT_IUPIrIGVM64_p1snb53LNZBhwZlbvHLaPz7PKpdv1SDm62gd96NJDVNS62cqybM49FUPn8Fjopx64ZDYIILxuNp7Xc37heph1n3X0KZcLcCADa7mGVEFjrjPYTdrtbcPDt9Fb5Y0NTgeKugSxOvsAJD5TJxx2b_CKQ8Qs5LpzYLoF4"
                      />
                      <div className="absolute top-1/4 right-1/4 glass-panel p-4 neon-glow-purple border-purple-500/30">
                        <div className="flex flex-col gap-2">
                          <div className="flex justify-between items-center gap-8">
                            <span className="font-['Space_Grotesk'] text-[10px] text-[#ddb7ff] font-bold">AGENT_BETA_9</span>
                            <span className="font-['Space_Grotesk'] text-[10px] text-[#ddb7ff] font-medium">SYNCED</span>
                          </div>
                          <div className="w-32 h-1 bg-[#2e3637] overflow-hidden">
                            <div className="w-3/4 h-full bg-[#ddb7ff]"></div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="absolute top-2 left-2 w-6 h-6 border-t-2 border-l-2 border-[#00dbe9]"></div>
                    <div className="absolute bottom-2 right-2 w-6 h-6 border-b-2 border-r-2 border-[#00dbe9]"></div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Live Intelligence Section */}
          <section className="px-12 mb-32">
            <div className="flex justify-between items-end mb-8">
              <div>
                <h2 className="font-['Space_Grotesk'] text-3xl font-semibold text-[#dbfcff]">Live Intelligence Feed</h2>
                <p className="text-[#b9cacb] font-['Inter'] mt-2">Real-time tactical data synthesis from active neural clusters.</p>
              </div>
              <div className="font-['Space_Grotesk'] text-xs text-[#dbfcff]/60">NODE_COUNT: 1,482 // LATENCY: 0.04MS</div>
            </div>
            <div className="glass-panel p-6 rounded-xl tactical-border neon-glow-cyan border-[#dbfcff]/20 min-h-[400px]">
              <div className="w-full h-full flex items-center justify-center text-[#00dbe9]/40 italic font-['Space_Grotesk']">
                {health ? `[ STREAMING LIVE DATA PACKETS... QUANTUM STABILITY: ${health.qubitStability || '98.42%'} ]` : '[ STREAMING LIVE DATA PACKETS... ]'}
              </div>
            </div>
          </section>

          {/* Research Pipeline Section */}
          <section className="px-12 mb-32">
            <h2 className="font-['Space_Grotesk'] text-3xl font-semibold text-[#dbfcff] mb-12">Deep Research Pipeline</h2>
            <div className="grid grid-cols-3 gap-6">
              {/* Experiment Success Rates */}
              <div className="glass-panel p-5 rounded-xl border-white/10 hover:border-[#dbfcff]/40 transition-colors group">
                <div className="flex justify-between items-start mb-8">
                  <span className="material-symbols-outlined text-[#00dbe9] text-3xl">analytics</span>
                  <span className="font-['Space_Grotesk'] text-[10px] text-[#b9cacb] uppercase">REF: EXP_084</span>
                </div>
                <h3 className="font-['Space_Grotesk'] text-2xl text-[#dce4e5] mb-4">Neural Training Success</h3>
                <div className="relative h-48 flex items-center justify-center">
                  <div className="w-32 h-32 rounded-full border-8 border-[#2e3637] relative flex items-center justify-center">
                    <div className="absolute inset-0 rounded-full border-8 border-[#dbfcff] border-t-transparent -rotate-45"></div>
                    <span className="font-['Space_Grotesk'] text-3xl text-[#dbfcff]">94.2%</span>
                  </div>
                </div>
                <p className="text-[#b9cacb] mt-4">Automated regression testing across 400 parameter sets completed with high-fidelity output.</p>
              </div>

              {/* Qubit Lattice Simulations */}
              <div className="glass-panel p-5 rounded-xl border-white/10 hover:border-[#ddb7ff]/40 transition-colors">
                <div className="flex justify-between items-start mb-8">
                  <span className="material-symbols-outlined text-[#ddb7ff] text-3xl">grid_view</span>
                  <span className="font-['Space_Grotesk'] text-[10px] text-[#b9cacb] uppercase">REF: QBT_LAT_9</span>
                </div>
                <h3 className="font-['Space_Grotesk'] text-2xl text-[#dce4e5] mb-4">Lattice Topology</h3>
                <div className="grid grid-cols-4 gap-2 h-48 content-center">
                  {[false, true, false, false, false, false, true, false].map((active, i) => (
                    <div key={i} className={`hex-grid ${active ? 'bg-[#ddb7ff] neon-glow-purple' : 'bg-[#6f00be]/40'} w-12 h-14 mx-auto`}></div>
                  ))}
                </div>
                <p className="text-[#b9cacb] mt-4">Real-time mapping of error correction thresholds in topological superconducting qubits.</p>
              </div>

              {/* Materials Discovery */}
              <div className="glass-panel p-5 rounded-xl border-white/10 hover:border-[#eac324]/40 transition-colors">
                <div className="flex justify-between items-start mb-8">
                  <span className="material-symbols-outlined text-[#eac324] text-3xl">matter</span>
                  <span className="font-['Space_Grotesk'] text-[10px] text-[#b9cacb] uppercase">REF: MAT_DISC</span>
                </div>
                <h3 className="font-['Space_Grotesk'] text-2xl text-[#dce4e5] mb-4">Synthesis Progress</h3>
                <div className="h-48 relative overflow-hidden rounded-lg">
                  <img
                    alt="Materials Discovery"
                    className="w-full h-full object-cover opacity-50 grayscale hover:grayscale-0 transition-all"
                    src="https://lh3.googleusercontent.com/aida-public/AB6AXuCvZM0_57whWr8aAFCpRzz3VjMq1wZiHFEqYE5bXi5BTbXYz_gvYFN--f0zmVT4zfj6minWdj-RRNXaJnYnRE4-lyujj4-p1KWVR_Cj_oLQNHr1uYDuIf-xYr6Cs6VjyqjF8BGmn23jPCUfCgJNUzWSGEUWT3aV1L5D4Y2XmdBta8JpQ0dZffyiynbkm-nenx4PxdnnhIJ9gizrDe0u0JzjP9s775z9tJdMnIquMRN3u0TQOonGgsSC22-wo7l7720CoAdIprIZJIY"
                  />
                  <div className="absolute bottom-2 left-2 bg-[#0d1515]/80 px-2 py-1 rounded text-[10px] font-['Space_Grotesk'] text-[#fff5de]">BONDING_STRENGTH: 98kJ/mol</div>
                </div>
                <p className="text-[#b9cacb] mt-4">Predictive modeling of high-temperature superconductors using neural density functional theory.</p>
              </div>
            </div>
          </section>

          {/* System Status Section */}
          <section className="px-12 mb-32">
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-4">
                <h2 className="font-['Space_Grotesk'] text-3xl font-semibold text-[#dbfcff] mb-4">Fleet Operations</h2>
                <p className="text-[#b9cacb] mb-6">Global system telemetry across all operational theaters. Monitor uptime, node stability, and deployment cycles in real-time.</p>
                <div className="space-y-4">
                  <div className="glass-panel p-4 rounded-lg flex items-center justify-between border-l-4 border-l-[#00dbe9]">
                    <span className="font-['Space_Grotesk'] text-[#dce4e5] font-bold text-xs">NORTH_AMERICA_HUB</span>
                    <span className="text-[#dbfcff] font-['Space_Grotesk']">{health?.uptime || '99.99%'}</span>
                  </div>
                  <div className="glass-panel p-4 rounded-lg flex items-center justify-between border-l-4 border-l-[#ddb7ff]">
                    <span className="font-['Space_Grotesk'] text-[#dce4e5] font-bold text-xs">EU_CENTRAL_ARRAY</span>
                    <span className="text-[#ddb7ff] font-['Space_Grotesk']">SYNCING</span>
                  </div>
                  <div className="glass-panel p-4 rounded-lg flex items-center justify-between border-l-4 border-l-[#ffb4ab]">
                    <span className="font-['Space_Grotesk'] text-[#dce4e5] font-bold text-xs">APAC_OFFSHORE_NODE</span>
                    <span className="text-[#ffb4ab] font-['Space_Grotesk']">OFFLINE</span>
                  </div>
                </div>
              </div>
              <div className="col-span-8">
                <div className="glass-panel h-full rounded-xl p-6 tactical-border neon-glow-purple border-[#ddb7ff]/20">
                  <div className="flex items-center gap-4 mb-4">
                    <span className="w-3 h-3 rounded-full bg-[#ddb7ff] animate-pulse"></span>
                    <span className="font-['Space_Grotesk'] text-[#ddb7ff] tracking-widest uppercase font-bold text-xs">Global Operational Status</span>
                  </div>
                  <div className="w-full h-[300px] flex items-center justify-center bg-[#080f10]/50 rounded-lg">
                    <div className="flex flex-col items-center gap-4 text-[#ddb7ff]/30">
                      <span className="material-symbols-outlined text-5xl">map</span>
                      <span className="font-['Space_Grotesk'] text-sm tracking-widest">[ ACCESSING GEOSPATIAL INTELLIGENCE... ]</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </main>

        {/* Footer Shell */}
        <footer className="glass-panel border-t border-[#3b494b]/20 bg-[#0d1515]/80 backdrop-blur-md">
          <div className="max-w-[1440px] mx-auto px-12 py-6">
            <div className="flex flex-col md:flex-row justify-between items-center gap-6">
              <div className="flex flex-col gap-2">
                <span className="font-['Space_Grotesk'] text-xs text-[#00dbe9] font-bold">QuLab AIOS</span>
                <p className="font-['Space_Grotesk'] text-[10px] text-[#b9cacb] uppercase tracking-widest">
                  © 2024 QULAB AIOS // TACTICAL INTELLIGENCE UNIT // COORD: 40.7128° N, 74.0060° W
                </p>
              </div>
              <div className="flex gap-8">
                <Link to="/labs" className="font-['Space_Grotesk'] text-xs uppercase tracking-widest text-[#b9cacb] hover:text-[#dbfcff] transition-colors">TERMS OF ENGAGEMENT</Link>
                <Link to="/system-lockdown" className="font-['Space_Grotesk'] text-xs uppercase tracking-widest text-[#b9cacb] hover:text-[#dbfcff] transition-colors">SECURITY PROTOCOL</Link>
                <Link to="/echo/integrations" className="font-['Space_Grotesk'] text-xs uppercase tracking-widest text-[#b9cacb] hover:text-[#dbfcff] transition-colors">NEURAL API</Link>
                <Link to="/screens" className="font-['Space_Grotesk'] text-xs uppercase tracking-widest text-[#b9cacb] hover:text-[#dbfcff] transition-colors">DATA SOVEREIGNTY</Link>
              </div>
            </div>
            <div className="mt-8 pt-8 border-t border-white/5 flex justify-between items-center opacity-30">
              <div className="font-['Space_Grotesk'] text-[10px] flex gap-4">
                <span>SYS_CORE: 0x88A2</span>
                <span>KERNEL: LUNAR_OS</span>
                <span>UPTIME: 421:12:09</span>
              </div>
              <div className="font-['Space_Grotesk'] text-[10px]">
                SECURED BY QULAB_ENCRYPTION_V2
              </div>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
