import { useLabHealth } from '../../lib/hooks';
import { useNavigate } from 'react-router';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function PharmacokineticsVisualizer() {
  const { health, loading } = useLabHealth('drug');
  const navigate = useNavigate();

  return (
    <div className="bg-[#101922] text-slate-200 font-['Space_Grotesk'] min-h-screen overflow-hidden flex flex-col relative">
      <style>{`
        ::-webkit-scrollbar {
          width: 4px;
        }
        ::-webkit-scrollbar-track {
          background: rgba(19, 127, 236, 0.05);
        }
        ::-webkit-scrollbar-thumb {
          background: rgba(19, 127, 236, 0.3);
          border-radius: 4px;
        }

        .fluid-bg {
          background: radial-gradient(circle at 50% 50%, rgba(19, 127, 236, 0.15) 0%, rgba(16, 25, 34, 0) 70%);
        }

        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .no-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}</style>

      {/* Background Simulation Layer */}
      <div className="absolute inset-0 z-0 overflow-hidden">
        <img
          alt="Abstract blue fluid dynamics simulation loop"
          className="w-full h-full object-cover opacity-80 mix-blend-overlay"
          src="https://lh3.googleusercontent.com/aida-public/AB6AXuCXcZv2KEm-6FWKL89HlB5mf-DVSBdh0vXoy3vmN98-_OHSBo01qE885wPRwuZaWII1gmuS8DK6HK5ekQM0fdGOdBIvdVYruETUIKMJc2pv-rgbwmgr4Q6aFY8pCyJk0JBthoWeXYxmsksaVAqAqUAqYcKr3X2BWQB2KjsqAw0Ld72AvlodKPKCRIsViWXERD0XxbrXf52_wooYgypU7qwZpJl3gDc0TQXA3Hdhp0EAlOQZJ9V45QAszEL0X0iFC6gj62xovSaLuus"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-[#101922]/90 via-transparent to-[#101922]/95"></div>
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-64 h-64 bg-[#137fec]/20 rounded-full blur-3xl animate-pulse"></div>
      </div>

      {/* Header Module */}
      <header className="relative z-10 pt-12 pb-4 px-6 flex items-center justify-between glass-panel border-b border-white/5">
        <div className="flex flex-col">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#137fec] animate-pulse shadow-[0_0_8px_rgba(19,127,236,0.8)]"></span>
            <span className="text-xs font-bold tracking-[0.2em] text-[#137fec] uppercase">Sim Active</span>
          </div>
          <h1 className="text-xl font-bold text-white tracking-wide mt-1">QULAB-8012</h1>
        </div>
        <div className="flex items-center gap-4 text-slate-400">
          <button
            onClick={() => navigate('/labs')}
            className="material-symbols-outlined text-[#137fec] text-sm hover:opacity-80"
          >
            arrow_back
          </button>
        </div>
      </header>

      {/* Main Content Area */}
            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="relative z-10 flex-1 flex flex-col px-4 pt-6 pb-24 overflow-y-auto no-scrollbar gap-6">
        {/* Simulation Context Label */}
        <div className="flex justify-between items-end">
          <div>
            <p className="text-xs uppercase tracking-widest text-slate-400 mb-1">Module Focus</p>
            <h2 className="text-lg text-white font-medium">Pharmacokinetics <br/> Absorption Loop</h2>
          </div>
          <button className="w-10 h-10 rounded-full glass-panel-light flex items-center justify-center text-[#137fec] active:scale-95 transition-transform">
            <span className="material-symbols-outlined">fullscreen</span>
          </button>
        </div>

        {/* Telemetry Grid */}
        <div className="grid grid-cols-2 gap-4">
          {/* Tmax Card */}
          <div className="glass-panel p-4 rounded-xl flex flex-col justify-between h-32 relative overflow-hidden group">
            <div className="absolute right-0 top-0 p-3 opacity-20 group-hover:opacity-40 transition-opacity">
              <span className="material-symbols-outlined text-[#137fec]">schedule</span>
            </div>
            <div>
              <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Tmax Horizon</span>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-3xl font-light text-white">{health?.tmax || '2.4'}</span>
                <span className="text-xs text-[#137fec] font-medium">hrs</span>
              </div>
            </div>
            <div className="w-full bg-slate-700/50 h-1 rounded-full mt-2 overflow-hidden">
              <div className="bg-[#137fec] h-full w-[65%] rounded-full shadow-[0_0_10px_rgba(19,127,236,0.6)]"></div>
            </div>
          </div>

          {/* Cmax Card */}
          <div className="glass-panel p-4 rounded-xl flex flex-col justify-between h-32 relative overflow-hidden">
            <div className="absolute right-0 top-0 p-3 opacity-20">
              <span className="material-symbols-outlined text-[#137fec]">analytics</span>
            </div>
            <div>
              <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Cmax Conc.</span>
              <div className="flex items-baseline gap-1 mt-1">
                <span className="text-3xl font-light text-white">450.2</span>
                <span className="text-xs text-[#137fec] font-medium">ng/mL</span>
              </div>
            </div>
            <div className="flex items-center gap-1 text-emerald-400 text-xs mt-2 bg-emerald-400/10 w-fit px-2 py-0.5 rounded-full border border-emerald-400/20">
              <span className="material-symbols-outlined text-[10px]">trending_up</span>
              <span>+4.2%</span>
            </div>
          </div>
        </div>

        {/* Echo's Insight Bubble */}
        <div className="glass-panel p-4 rounded-xl border-l-4 border-l-[#137fec] flex gap-4 items-start shadow-lg">
          <div className="mt-1 flex-shrink-0">
            <div className="w-8 h-8 rounded-full bg-[#137fec]/20 flex items-center justify-center border border-[#137fec]/30">
              <span className="material-symbols-outlined text-[#137fec] text-sm animate-pulse">auto_awesome</span>
            </div>
          </div>
          <div>
            <p className="text-xs font-bold text-[#137fec] mb-1 uppercase tracking-wider">Echo Insight</p>
            <p className="text-sm text-slate-200 leading-relaxed font-light">
              Predicting metabolic pathway clearance. Logic grounded in verified clinical models.
            </p>
          </div>
        </div>

        {/* Secondary Metrics List */}
        <div className="glass-panel rounded-xl overflow-hidden divide-y divide-white/5">
          <div className="p-4 flex justify-between items-center">
            <div className="flex items-center gap-3">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-500"></span>
              <span className="text-sm text-slate-300">Bioavailability (F)</span>
            </div>
            <span className="text-sm font-mono text-white">{health?.bioavailability || '82.4%'}</span>
          </div>
          <div className="p-4 flex justify-between items-center">
            <div className="flex items-center gap-3">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-500"></span>
              <span className="text-sm text-slate-300">Volume of Dist. (Vd)</span>
            </div>
            <span className="text-sm font-mono text-white">4.2 L/kg</span>
          </div>
          <div className="p-4 flex justify-between items-center">
            <div className="flex items-center gap-3">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-500"></span>
              <span className="text-sm text-slate-300">Clearance (CL)</span>
            </div>
            <span className="text-sm font-mono text-white">12 mL/min</span>
          </div>
        </div>
      </main>

      {/* Bottom Action Bar */}
      <div className="fixed bottom-0 left-0 right-0 p-4 z-20 bg-gradient-to-t from-[#101922] via-[#101922]/95 to-transparent">
        <div className="flex items-center gap-3">
          <button className="flex-1 h-14 bg-white text-[#101922] rounded-xl font-bold flex items-center justify-center gap-2 hover:bg-slate-200 transition-colors shadow-[0_0_20px_rgba(255,255,255,0.1)]">
            <span className="material-symbols-outlined">play_arrow</span>
            <span>Inject</span>
          </button>
          <button className="w-14 h-14 glass-panel rounded-xl flex items-center justify-center text-white hover:bg-white/10 transition-colors active:scale-95">
            <span className="material-symbols-outlined">pause</span>
          </button>
          <button className="w-14 h-14 glass-panel rounded-xl flex items-center justify-center text-[#137fec] hover:bg-white/10 transition-colors active:scale-95 border border-[#137fec]/30">
            <span className="material-symbols-outlined">science</span>
          </button>
        </div>
        <div className="h-1 w-1/3 bg-slate-600 rounded-full mx-auto mt-4"></div>
      </div>
    </div>
  );
}
