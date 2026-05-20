import { useLabsHealth, useLabsConfig } from '../../lib/hooks';
import { getLabRoute } from '../../lib/lab-routes';
import { Link } from 'react-router';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function LiveLabWallFigma() {
  const { labsStatus, loading } = useLabsHealth();
  const { labs } = useLabsConfig();

  const labsArray = Object.entries(labs).map(([key, config]) => ({
    key,
    ...config,
    status: labsStatus[key],
  }));

  const activeLabs = labsArray.filter(lab => lab.status?.healthy);
  const inactiveLabs = labsArray.filter(lab => !lab.status?.healthy);

  return (
    <div className="bg-[#101922] font-['Space_Grotesk'] text-slate-100 antialiased overflow-x-hidden">
      <style>{`
        .glow-primary {
          box-shadow: 0 0 15px rgba(19, 127, 236, 0.3);
        }
        .text-glow {
          text-shadow: 0 0 8px rgba(19, 127, 236, 0.6);
        }
      `}</style>

      <div className="relative flex min-h-screen w-full flex-col pb-32">
        {/* Top HUD Header */}
        <header className="sticky top-0 z-50 flex items-center justify-between p-4 glass-dark border-b border-white/5">
          <Link to="/" className="flex items-center gap-3">
            <div className="bg-[#137fec]/20 p-2 rounded-lg border border-[#137fec]/30">
              <span className="material-symbols-outlined text-[#137fec] text-2xl">rocket_launch</span>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">QulabInfinite</h1>
              <p className="text-[10px] uppercase tracking-[0.2em] text-[#137fec]/80 font-semibold leading-none">
                Fleet Operations
              </p>
            </div>
          </Link>
          <div className="flex items-center gap-4">
            <div className="text-right hidden sm:block">
              <p className="text-[10px] text-slate-400 uppercase tracking-wider">System Sync</p>
              <p className="text-xs font-mono">0.002ms LAG</p>
            </div>
            <button className="relative glass p-2 rounded-full hover:bg-white/10 transition-colors">
              <span className="material-symbols-outlined text-slate-100">notifications</span>
              <span className="absolute top-1 right-1 size-2 bg-[#137fec] rounded-full glow-primary"></span>
            </button>
          </div>
        </header>

              <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="flex-1 px-4 py-6 space-y-8">
          {/* Active Production Fleet Section */}
          <section>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className="size-2 bg-green-500 rounded-full animate-pulse"></span>
                <h2 className="text-xl font-bold">Active Production Fleet</h2>
              </div>
              <span className="text-xs font-medium text-slate-400">{activeLabs.length} Labs Online</span>
            </div>

            {/* Active Cards Horizontal Scroll */}
            <div className="flex gap-4 overflow-x-auto pb-4 snap-x snap-mandatory">
              {activeLabs.map((lab) => (
                <Link
                  key={lab.key}
                  to={getLabRoute(lab.key)}
                  className="relative flex-shrink-0 w-[85vw] max-w-sm aspect-[4/5] rounded-xl overflow-hidden snap-center border border-white/10 group"
                >
                  <div className="absolute inset-0 bg-gradient-to-t from-[#101922] via-transparent to-transparent z-10"></div>
                  <div className="absolute inset-0 w-full h-full bg-gradient-to-br from-[#137fec]/20 to-[#6f00be]/20 group-hover:opacity-80 transition-opacity duration-700"></div>

                  {/* Floating Badge */}
                  <div className="absolute top-4 left-4 z-20 glass px-3 py-1 rounded-full flex items-center gap-2">
                    <span className="material-symbols-outlined text-amber-400 text-sm">verified</span>
                    <span className="text-[10px] font-bold tracking-widest uppercase">Online</span>
                  </div>

                  {/* Data Overlay */}
                  <div className="absolute bottom-0 left-0 right-0 p-5 z-20 space-y-3">
                    <div className="flex justify-between items-end">
                      <div>
                        <h3 className="text-2xl font-bold">{lab.name}</h3>
                        <p className="text-sm text-slate-300">
                          {lab.type === 'unified' ? 'Unified API' : `Port ${lab.port}`}
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-[10px] text-[#137fec] font-bold uppercase tracking-tighter">STATUS</p>
                        <p className="text-lg font-mono leading-none text-green-400">
                          {lab.status?.status || 'ONLINE'}
                        </p>
                      </div>
                    </div>
                    <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full bg-[#137fec] w-[98%] glow-primary"></div>
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          </section>

          {/* Quarantine & Future R&D Section */}
          <section>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <span className="size-2 bg-amber-500 rounded-full"></span>
                <h2 className="text-xl font-bold">Offline Labs &amp; Maintenance</h2>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4">
              {inactiveLabs.map((lab) => (
                <div
                  key={lab.key}
                  className="glass p-4 rounded-xl space-y-4 opacity-80 border-l-4 border-l-amber-500/50"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-bold text-slate-100">{lab.name}</h4>
                      <p className="text-xs text-slate-400">
                        {lab.type === 'unified' ? 'Unified API' : `Port ${lab.port}`}
                      </p>
                    </div>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-500 border border-amber-500/20">
                      OFFLINE
                    </span>
                  </div>
                  <div className="space-y-2">
                    <p className="text-[10px] uppercase tracking-wider font-semibold text-slate-500">Status</p>
                    <ul className="space-y-1.5">
                      <li className="flex items-center gap-2 text-xs text-slate-300">
                        <span className="material-symbols-outlined text-sm text-amber-500/70">warning</span>
                        Awaiting Connection
                      </li>
                    </ul>
                  </div>
                </div>
              ))}
            </div>
          </section>
        </main>

        {/* Echo Floating Orb & Summary */}
        <div className="fixed bottom-24 left-1/2 -translate-x-1/2 z-[60] w-[90%] max-w-md">
          <div className="glass p-4 rounded-2xl flex items-center gap-4 shadow-2xl border border-white/10">
            <div className="relative shrink-0">
              <div className="absolute -inset-1 bg-[#137fec]/30 rounded-full blur-sm animate-pulse"></div>
              <div className="relative size-12 rounded-full overflow-hidden border-2 border-[#137fec] bg-[#101922] flex items-center justify-center">
                <span className="material-symbols-outlined text-[#137fec] text-2xl">psychology</span>
              </div>
            </div>
            <div className="space-y-0.5">
              <p className="text-[10px] font-bold text-[#137fec] uppercase tracking-widest leading-none">
                Echo Overview
              </p>
              <p className="text-xs text-slate-200 leading-tight">
                "We have {activeLabs.length} labs at peak production; {inactiveLabs.length} offline or in
                testing."
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Navigation Bar */}
        <nav className="fixed bottom-4 left-4 right-4 z-[70] h-16 glass rounded-2xl flex items-center justify-around px-2 border border-white/10">
          <Link to="/labs" className="flex flex-col items-center gap-1 text-[#137fec]">
            <span className="material-symbols-outlined fill-1">dashboard</span>
            <span className="text-[10px] font-medium">Fleet</span>
          </Link>
          <Link to="/intel-dashboard" className="flex flex-col items-center gap-1 text-slate-400 hover:text-[#137fec]">
            <span className="material-symbols-outlined">analytics</span>
            <span className="text-[10px] font-medium">Analytics</span>
          </Link>
          <Link to="/screens" className="size-10 bg-[#137fec] rounded-full flex items-center justify-center glow-primary -mt-8 shadow-lg">
            <span className="material-symbols-outlined text-white">add</span>
          </Link>
          <Link to="/labs/materials" className="flex flex-col items-center gap-1 text-slate-400 hover:text-[#137fec]">
            <span className="material-symbols-outlined">science</span>
            <span className="text-[10px] font-medium">R&amp;D</span>
          </Link>
          <Link to="/" className="flex flex-col items-center gap-1 text-slate-400">
            <span className="material-symbols-outlined">settings</span>
            <span className="text-[10px] font-medium">Home</span>
          </Link>
        </nav>
      </div>
    </div>
  );
}
