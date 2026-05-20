import { useLabHealth } from '../../lib/hooks';
import { useNavigate } from 'react-router';
import { Image3DViewer } from '../components/Image3DViewer';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function GenomicsLabVisualizer() {
  const { health, loading } = useLabHealth('genomics');
  const navigate = useNavigate();

  return (
    <div className="bg-[#101922] text-white font-['Space_Grotesk'] antialiased h-screen overflow-hidden relative">
      <style>{`
        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .no-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
        .glass-card {
          background: rgba(19, 127, 236, 0.1);
          backdrop-filter: blur(8px);
          -webkit-backdrop-filter: blur(8px);
          border: 1px solid rgba(19, 127, 236, 0.2);
        }
        @keyframes pulse-glow {
          0%, 100% { opacity: 1; box-shadow: 0 0 5px #137fec; }
          50% { opacity: 0.6; box-shadow: 0 0 15px #137fec; }
        }
        .animate-pulse-glow {
          animation: pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        .text-gradient {
          background-clip: text;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-image: linear-gradient(to right, #137fec, #60a5fa);
        }
      `}</style>

      {/* Background 3D Visualizer */}
      <div className="absolute inset-0 z-0 bg-[#101922]">
        <Image3DViewer
          imageUrl="https://lh3.googleusercontent.com/aida-public/AB6AXuDdcmP2Ey6-8UezmXWkkA4ykpypK5CYV8X4WFG3GxypgkgITnMShAoIU-OWtybG-aSMWxJFU1pLFfTe_WLrUKAtvlEIOanc61zyxRH0FNB7WCGCvkpkMFqFEvD_lKy6azOssWzbCOX0iwJ8pW0p9vF1QTDT4nkfSOGYj2Zni_HcngfedR-ljGqDdyJi1Gqe1keh2Xa-j-c9hd6z0Id4R9-UZX9qKn-4n9fqa5lOh4uUPEFDrov6S24me8zs1EauNps5Tj2xachO74c"
          alt="DNA Helix 3D Visualization"
          className="absolute inset-0 w-full h-full opacity-80"
          autoRotate={true}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-[#101922]/80 via-transparent to-[#101922]"></div>
        <div className="absolute inset-0 bg-gradient-to-r from-[#101922]/40 via-transparent to-[#101922]/40"></div>
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage:
              'linear-gradient(#137fec 1px, transparent 1px), linear-gradient(90deg, #137fec 1px, transparent 1px)',
            backgroundSize: '40px 40px',
          }}
        ></div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 flex flex-col h-full">
        {/* Top Navigation */}
        <header className="flex items-center justify-between px-6 pt-12 pb-4 w-full">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/labs')}
              className="w-10 h-10 flex items-center justify-center rounded-full glass-panel hover:bg-white/10 transition-colors"
            >
              <span className="material-symbols-outlined text-white/80">arrow_back</span>
            </button>
            <div className="flex flex-col">
              <h1 className="text-sm font-bold tracking-wider uppercase text-[#137fec]">QulabInfinite</h1>
              <span className="text-xs text-slate-400">Genomics Lab // Unified API</span>
            </div>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full glass-panel border-[#137fec]/30">
            <span className="w-2 h-2 rounded-full bg-[#137fec] animate-pulse-glow"></span>
            <span className="text-xs font-semibold text-[#137fec] tracking-widest uppercase">
              {health ? 'Live' : 'Offline'}
            </span>
          </div>
        </header>

        {/* Central Viewport Markers */}
        <div className="flex-1 relative p-6">
          <div className="absolute top-1/4 left-1/4 flex items-center gap-2 opacity-80">
            <div className="w-3 h-3 rounded-full border border-white/50 bg-[#137fec]/20"></div>
            <div className="h-[1px] w-12 bg-gradient-to-r from-white/50 to-transparent"></div>
            <span className="text-[10px] bg-black/40 px-1 rounded text-[#137fec]">Seq. A-T</span>
          </div>
          <div className="absolute top-1/3 right-8 flex flex-col items-end gap-1 opacity-90">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-bold text-[#ec137f] uppercase tracking-widest">
                Mutation Detect
              </span>
              <div className="w-2 h-2 rounded-full bg-[#ec137f] animate-ping"></div>
            </div>
            <div className="text-[10px] text-slate-300 glass-panel px-2 py-1 rounded">Locus: 12p13.31</div>
          </div>

          {/* Central HUD Overlay */}
          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 border border-white/10 rounded-full flex items-center justify-center pointer-events-none">
            <div className="w-56 h-56 border border-[#137fec]/20 rounded-full border-dashed animate-[spin_10s_linear_infinite]"></div>
            <div className="absolute top-0 w-1 h-4 bg-[#137fec]/50"></div>
            <div className="absolute bottom-0 w-1 h-4 bg-[#137fec]/50"></div>
            <div className="absolute left-0 h-1 w-4 bg-[#137fec]/50"></div>
            <div className="absolute right-0 h-1 w-4 bg-[#137fec]/50"></div>
          </div>
        </div>

        {/* Bottom Sheet */}
        <div className="mt-auto w-full px-4 pb-6">
          <div className="glass-panel rounded-3xl p-5 w-full shadow-2xl shadow-black/50 border-t border-white/10">
            {/* Metrics Row */}
            <div className="flex gap-3 mb-6 overflow-x-auto no-scrollbar pb-2">
              <div className="min-w-[140px] glass-card p-3 rounded-xl flex flex-col relative overflow-hidden group">
                <div className="absolute -right-4 -top-4 w-12 h-12 bg-[#137fec]/20 blur-xl rounded-full"></div>
                <div className="flex justify-between items-start mb-2">
                  <span className="text-[10px] font-medium text-slate-300 uppercase tracking-wide">Seq Depth</span>
                  <span className="material-symbols-outlined text-[#137fec] text-sm">show_chart</span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-white">142x</span>
                  <span className="text-[10px] text-green-400">↑ 12%</span>
                </div>
                <div className="w-full h-1 bg-white/10 mt-3 rounded-full overflow-hidden">
                  <div className="h-full bg-[#137fec] w-3/4"></div>
                </div>
              </div>

              <div className="min-w-[140px] bg-white/5 border border-white/5 p-3 rounded-xl flex flex-col relative overflow-hidden">
                <div className="flex justify-between items-start mb-2">
                  <span className="text-[10px] font-medium text-slate-300 uppercase tracking-wide">Error Rate</span>
                  <span className="material-symbols-outlined text-slate-400 text-sm">
                    history_toggle_off
                  </span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-white">
                    0.004<span className="text-sm text-slate-400">%</span>
                  </span>
                </div>
                <svg className="w-full h-4 mt-2 opacity-50" preserveAspectRatio="none" viewBox="0 0 100 20">
                  <path
                    className="text-green-400"
                    d="M0 15 Q 10 18, 20 10 T 40 12 T 60 5 T 80 8 T 100 15"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  ></path>
                </svg>
              </div>

              <div className="min-w-[140px] bg-white/5 border border-white/5 p-3 rounded-xl flex flex-col">
                <div className="flex justify-between items-start mb-2">
                  <span className="text-[10px] font-medium text-slate-300 uppercase tracking-wide">Reads</span>
                  <span className="material-symbols-outlined text-slate-400 text-sm">data_object</span>
                </div>
                <div className="flex items-baseline gap-1">
                  <span className="text-2xl font-bold text-white">4.2M</span>
                  <span className="text-[10px] text-slate-400">bp</span>
                </div>
                <div className="flex gap-1 mt-3">
                  <div className="h-1 w-1 rounded-full bg-[#137fec] animate-pulse"></div>
                  <div className="h-1 w-1 rounded-full bg-[#137fec]/50 animate-pulse delay-75"></div>
                  <div className="h-1 w-1 rounded-full bg-[#137fec]/20 animate-pulse delay-150"></div>
                </div>
              </div>
            </div>

            {/* Echo's Message */}
            <div className="flex items-start gap-3 mb-6 bg-gradient-to-r from-[#137fec]/10 to-transparent p-3 rounded-xl border-l-2 border-[#137fec]">
              <div className="relative shrink-0">
                <div className="w-10 h-10 rounded-full overflow-hidden border border-white/20 flex items-center justify-center bg-[#137fec]/20">
                  <span className="material-symbols-outlined text-[#137fec]">psychology</span>
                </div>
                <div className="absolute -bottom-1 -right-1 bg-[#101922] p-0.5 rounded-full">
                  <span className="block w-2.5 h-2.5 bg-green-500 rounded-full border border-[#101922]"></span>
                </div>
              </div>
              <div className="flex flex-col">
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-xs font-bold text-white">Echo</span>
                  <span className="text-[10px] px-1 rounded bg-white/10 text-slate-400">Orchestrator</span>
                </div>
                <p className="text-sm text-slate-200 leading-snug">
                  Alignment with reference genome{' '}
                  <span className="text-[#137fec] font-mono text-xs">GRCh38</span> complete.
                  NIST-traceable accuracy confirmed.
                </p>
              </div>
            </div>

            {/* Action Controls */}
            <div className="grid grid-cols-4 gap-3">
              <button className="col-span-1 flex flex-col items-center justify-center gap-1 p-3 rounded-xl bg-white/5 hover:bg-white/10 transition-all border border-transparent hover:border-white/10 group">
                <span className="material-symbols-outlined text-white group-hover:text-[#137fec] transition-colors">
                  pause
                </span>
                <span className="text-[10px] font-medium text-slate-400">Pause</span>
              </button>
              <button className="col-span-1 flex flex-col items-center justify-center gap-1 p-3 rounded-xl bg-white/5 hover:bg-white/10 transition-all border border-transparent hover:border-white/10 group">
                <span className="material-symbols-outlined text-white group-hover:text-[#137fec] transition-colors">
                  analytics
                </span>
                <span className="text-[10px] font-medium text-slate-400">Analyze</span>
              </button>
              <button className="col-span-1 flex flex-col items-center justify-center gap-1 p-3 rounded-xl bg-white/5 hover:bg-white/10 transition-all border border-transparent hover:border-white/10 group">
                <span className="material-symbols-outlined text-white group-hover:text-[#137fec] transition-colors">
                  download
                </span>
                <span className="text-[10px] font-medium text-slate-400">Export</span>
              </button>
              <button className="col-span-1 flex flex-col items-center justify-center gap-1 p-3 rounded-xl bg-[#137fec] hover:bg-[#0b5cb5] transition-all shadow-[0_0_15px_rgba(19,127,236,0.4)]">
                <span className="material-symbols-outlined text-white">science</span>
                <span className="text-[10px] font-bold text-white">Full View</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
