import { Link } from 'react-router';
import { Navigation } from '../components/Navigation';
import { useLabsHealth, useLabsConfig } from '../../lib/hooks';
import { STITCH_HERO_SCREENS } from '../../lib/lab-routes';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function MissionControl() {
  const { labsStatus, loading } = useLabsHealth();
  const { labs } = useLabsConfig();

  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 overflow-hidden">
          <EchoLabCommandInline className="mb-8" />
        {/* Hero Section */}
        <section className="px-[48px] relative mb-32">
          <div className="grid grid-cols-12 gap-6 items-center">
            <div className="col-span-5 z-10">
              <div className="mb-4 inline-flex items-center gap-2 px-3 py-1 bg-[rgba(111,0,190,0.2)] border border-[rgba(111,0,190,0.4)] rounded-full">
                <span className="w-2 h-2 rounded-full bg-[#ddb7ff] animate-pulse"></span>
                <span className="text-[12px] leading-[16px] tracking-[0.15em] font-bold text-[#ddb7ff]">QULAB_INFINITE_V1</span>
              </div>
              <h1 className="text-[48px] leading-[56px] tracking-[-0.02em] font-bold text-[#dbfcff] mb-6">
                Advanced R&D <br />
                <span className="text-[#ddb7ff]">Computational Labs</span>
              </h1>
              <p className="text-[18px] leading-[28px] text-[#b9cacb] mb-8 max-w-xl">
                Materials science, quantum chemistry, digital twin simulation, and diagnostic intelligence. 13 production-grade computational engines powered by validated algorithms and real-world physics.
              </p>
              <div className="flex gap-4">
                <Link to="/labs" className="px-8 py-4 bg-[#00dbe9] text-[#0d1515] text-[12px] leading-[16px] tracking-[0.15em] font-bold hover:brightness-110 transition-all tactical-border uppercase">
                  VIEW ALL LABS
                </Link>
              </div>
            </div>
            <div className="col-span-7 relative h-[600px]">
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="relative w-full h-full glass-panel neon-glow-cyan border-white/5 rounded-xl overflow-hidden p-6">
                  <h3 className="text-[20px] font-semibold text-[#00dbe9] mb-4">Live Lab Status</h3>
                  <div className="space-y-3 overflow-auto max-h-[500px]">
                    {loading ? (
                      <div className="text-[#b9cacb]">Loading labs...</div>
                    ) : (
                      Object.entries(labsStatus).map(([labKey, status]) => (
                        <div key={labKey} className="flex items-center justify-between p-3 bg-[#192122] rounded-lg">
                          <span className="text-[#dce4e5]">{labs[labKey].name}</span>
                          <div className="flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${status.healthy ? 'bg-[#00dbe9]' : 'bg-[#ffb4ab]'} animate-pulse`}></span>
                            <span className={`text-[12px] ${status.healthy ? 'text-[#00dbe9]' : 'text-[#ffb4ab]'}`}>
                              {status.healthy ? 'ONLINE' : 'OFFLINE'}
                            </span>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Labs Overview */}
        <section className="px-[48px] mb-32">
          <h2 className="text-[32px] leading-[40px] tracking-[-0.01em] font-semibold text-[#dbfcff] mb-12">Research & Development Labs</h2>
          <div className="grid grid-cols-3 gap-6">
            <div className="glass-panel p-6 rounded-xl border-white/10 hover:border-[#ddb7ff]/40 transition-colors">
              <h3 className="text-[20px] font-semibold text-[#ddb7ff] mb-4">Materials & Chemistry</h3>
              <div className="text-[48px] font-bold text-[#dbfcff]">3</div>
              <p className="text-[#b9cacb] mt-2">Advanced computational R&D engines</p>
            </div>
            <div className="glass-panel p-6 rounded-xl border-white/10 hover:border-[#00dbe9]/40 transition-colors">
              <h3 className="text-[20px] font-semibold text-[#00dbe9] mb-4">Total Labs</h3>
              <div className="text-[48px] font-bold text-[#dbfcff]">{Object.keys(labs).length}</div>
              <p className="text-[#b9cacb] mt-2">Production-ready computational systems</p>
            </div>
            <div className="glass-panel p-6 rounded-xl border-white/10 hover:border-[#00f0ff]/40 transition-colors">
              <h3 className="text-[20px] font-semibold text-[#00f0ff] mb-4">Labs Online</h3>
              <div className="text-[48px] font-bold text-[#dbfcff]">
                {Object.values(labsStatus).filter(s => s.healthy).length}
              </div>
              <p className="text-[#b9cacb] mt-2">Active computational endpoints</p>
            </div>
          </div>
        </section>

        <section className="px-[48px] mb-24">
          <div className="flex items-end justify-between mb-8">
            <h2 className="text-[32px] leading-[40px] font-semibold text-[#dbfcff]">Stitch hero screens</h2>
            <Link to="/screens" className="text-[12px] tracking-[0.15em] font-bold text-[#00dbe9] uppercase hover:underline">
              View all
            </Link>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {STITCH_HERO_SCREENS.map((screen) => (
              <Link
                key={screen.id}
                to={screen.path}
                className="glass-panel p-4 rounded-xl border border-white/10 hover:border-[#00dbe9]/40 transition-colors"
              >
                <span className="material-symbols-outlined text-[#00dbe9]">{screen.icon}</span>
                <h3 className="text-[16px] font-semibold text-[#dbfcff] mt-2">{screen.title}</h3>
                <p className="text-[12px] text-[#849495]">{screen.subtitle}</p>
              </Link>
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="glass-panel border-t border-[#3b494b]/20 bg-[#0d1515]/80 backdrop-blur-md">
        <div className="max-w-[1440px] mx-auto px-[48px] py-6">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="flex flex-col gap-2">
              <span className="text-[12px] tracking-[0.15em] font-bold text-[#00dbe9]">QuLab AIOS</span>
              <p className="text-[10px] text-[#b9cacb] uppercase tracking-widest">
                © 2024 QULAB AIOS // TACTICAL INTELLIGENCE UNIT
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
