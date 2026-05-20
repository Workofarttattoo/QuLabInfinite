import { useLabHealth } from '../../lib/hooks';
import { useNavigate } from 'react-router';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function GenomicsProductionVisualizer() {
  const { health, loading } = useLabHealth('genomics');
  const navigate = useNavigate();

  return (
    <div className="bg-[#080c11] text-slate-300 font-['Space_Grotesk'] antialiased h-screen overflow-hidden relative">
      <style>{`
        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .no-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }

        .glass-card {
          background: rgba(255, 255, 255, 0.03);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          border: 0.5px solid rgba(255, 255, 255, 0.05);
        }

        .micro-typo {
          font-size: 0.6rem;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          font-weight: 600;
        }

        .nist-badge {
          background: linear-gradient(90deg, rgba(0, 245, 160, 0.13), transparent);
          border-left: 2px solid #00f5a0;
        }

        .grounded-node {
          box-shadow: 0 0 15px rgba(19, 127, 236, 0.4);
        }
      `}</style>

      <div className="absolute inset-0 z-0 overflow-hidden">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-60 scale-110"
          style={{
            backgroundImage: "url('https://lh3.googleusercontent.com/aida-public/AB6AXuDdcmP2Ey6-8UezmXWkkA4ykpypK5CYV8X4WFG3GxypgkgITnMShAoIU-OWtybG-aSMWxJFU1pLFfTe_WLrUKAtvlEIOanc61zyxRH0FNB7WCGCvkpkMFqFEvD_lKy6azOssWzbCOX0iwJ8pW0p9vF1QTDT4nkfSOGYj2Zni_HcngfedR-ljGqDdyJi1Gqe1keh2Xa-j-c9hd6z0Id4R9-UZX9qKn-4n9fqa5lOh4uUPEFDrov6S24me8zs1EauNps5Tj2xachO74c')"
          }}
        ></div>
        <div className="absolute inset-0 bg-gradient-to-b from-[#080c11] via-[#080c11]/20 to-[#080c11]"></div>
        <div className="absolute top-[40%] left-[30%] w-6 h-6 rounded-full border border-[#137fec]/40 flex items-center justify-center animate-pulse">
          <div className="w-1.5 h-1.5 bg-[#137fec] rounded-full grounded-node"></div>
        </div>
        <div className="absolute top-[55%] right-[25%] w-6 h-6 rounded-full border border-[#ec137f]/40 flex items-center justify-center">
          <div className="w-1.5 h-1.5 bg-[#ec137f] rounded-full grounded-node"></div>
        </div>
      </div>

      <div className="relative z-10 flex flex-col h-full safe-area-inset-top">
        <header className="flex items-center justify-between px-5 pt-10 pb-4">
          <div className="flex flex-col">
            <div className="flex items-center gap-1.5">
              <span className="w-1 h-3 bg-[#137fec] rounded-full"></span>
              <h1 className="text-xs font-bold tracking-[0.2em] text-white uppercase">Refined Genomics Suite</h1>
            </div>
            <div className="flex items-center gap-2 mt-1">
              <span className="micro-typo text-slate-500">Node ID: QULAB-8015</span>
              <span className="w-1 h-1 bg-slate-700 rounded-full"></span>
              <span className="micro-typo text-[#137fec]">Active Sequence</span>
            </div>
          </div>
          <button
            onClick={() => navigate('/labs')}
            className="w-9 h-9 rounded-full glass-panel flex items-center justify-center"
          >
            <span className="material-symbols-outlined text-[20px] text-white">grid_view</span>
          </button>
        </header>

        <div className="flex-1 relative">
          <div className="absolute top-4 left-5 nist-badge px-3 py-2 flex flex-col gap-0.5">
            <span className="micro-typo text-[#00f5a0]">NIST TRACEABLE</span>
            <span className="text-[10px] text-slate-400 font-mono">ACC: {health?.accuracy || '99.999984%'}</span>
          </div>
          <div className="absolute top-1/2 left-[35%] -translate-y-1/2 glass-panel px-3 py-2 rounded-lg border-l-2 border-[#137fec]">
            <div className="flex items-center gap-2 mb-1">
              <span className="micro-typo text-[#137fec]">Grounded Node</span>
              <span className="text-[8px] font-mono text-slate-500">#44-AT</span>
            </div>
            <div className="text-[11px] text-white font-medium">Mutation Detected</div>
            <div className="text-[9px] text-slate-400 mt-1">Locus: 12p13.31 • SNP</div>
          </div>
        </div>

        <div className="mt-auto px-4 pb-6 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="glass-panel rounded-2xl p-4 flex flex-col justify-between">
              <div>
                <span className="micro-typo text-slate-500">Base-Pair Telemetry</span>
                <div className="flex items-baseline gap-1 mt-1">
                  <span className="text-2xl font-bold text-white tracking-tight">{health?.basePairRate || '4.21'}</span>
                  <span className="text-[10px] text-[#137fec] font-mono">Tbp/h</span>
                </div>
              </div>
              <div className="flex gap-1 mt-3">
                <div className="h-[2px] flex-1 bg-[#137fec]/20 rounded-full overflow-hidden">
                  <div className="h-full bg-[#137fec] w-2/3"></div>
                </div>
              </div>
            </div>
            <div className="glass-panel rounded-2xl p-4 flex flex-col justify-between border-t border-white/10">
              <div className="flex justify-between items-start">
                <span className="micro-typo text-slate-500">Reagent Level</span>
                <span className="material-symbols-outlined text-[14px] text-[#ec137f]">opacity</span>
              </div>
              <div className="mt-2">
                <div className="text-lg font-bold text-white">18.4%</div>
                <div className="flex items-center gap-1.5 mt-1">
                  <div className="flex-1 h-1 bg-white/5 rounded-full">
                    <div className="h-full bg-[#ec137f] w-[18.4%]"></div>
                  </div>
                  <span className="text-[8px] font-mono text-slate-500">LOW</span>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-panel rounded-2xl p-4 border-l-2 border-[#137fec] overflow-hidden relative">
            <div className="absolute -right-4 -top-4 w-16 h-16 bg-[#137fec]/5 blur-2xl rounded-full"></div>
            <div className="flex items-center gap-3 mb-3">
              <div className="relative">
                <img
                  alt="Echo"
                  className="w-10 h-10 rounded-full object-cover grayscale opacity-80 border border-white/20 shadow-lg"
                  src="https://lh3.googleusercontent.com/aida-public/AB6AXuAqRQ2OW-ybA8nw7l0qnUF5N-rdCOO2_-pCfnvgrgXrkIopvKbYzxqIBS_KAgTsicDCuUBTgleYL8S93HqEVe7tAHdlb_Sa0dRVyiNMVDKkJ8HqpqNqqu6mvJbInwJ2qjoUIGL1e64Wo_hEmy8WCnVHvrDUl-U01Tp9PPjLf1nAmpDeeeqWuX5lacE9jW4Op3T9mg5bYIvGgKI2VGRg5CJGRr0exPGGx4AyIFxHqbSKrvdN8OT_DfYhrlx3HQfZnj-oP6MxPca2cFA"
                />
                <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-[#00f5a0] rounded-full border-2 border-[#080c11]"></div>
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-white tracking-wide">Echo AGI</span>
                  <span className="micro-typo text-[#137fec] bg-[#137fec]/10 px-1 rounded">Orchestrator</span>
                </div>
                <span className="text-[9px] text-slate-500 font-mono italic">Reasoning: Multimodal Heuristics Active</span>
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-xs text-slate-300 leading-relaxed font-light">
                Reference <span className="text-[#137fec] font-mono text-[10px]">GRCh38</span> verification sync successful. Signal-to-noise ratio within clinical parameters.
              </p>
              <div className="bg-white/5 rounded-lg p-2.5 border border-white/5">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <span className="material-symbols-outlined text-[14px] text-[#00f5a0]">bolt</span>
                  <span className="micro-typo text-[#00f5a0]">Next-Best-Action</span>
                </div>
                <p className="text-[11px] text-slate-200">Initiate automated buffer injection to counteract reagent depletion and stabilize throughput drift.</p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button className="flex-1 bg-white/5 hover:bg-white/10 h-12 rounded-xl flex items-center justify-center gap-2 transition-all group border border-white/5">
              <span className="material-symbols-outlined text-[18px] text-slate-400 group-hover:text-white">tune</span>
              <span className="micro-typo text-slate-400 group-hover:text-white">Parameters</span>
            </button>
            <button className="flex-[1.5] bg-[#137fec] h-12 rounded-xl flex items-center justify-center gap-2 shadow-lg shadow-[#137fec]/20">
              <span className="material-symbols-outlined text-[18px] text-white">rocket_launch</span>
              <span className="micro-typo text-white font-bold">Execute Ops</span>
            </button>
          </div>
        </div>
      </div>

      <div className="fixed inset-0 pointer-events-none z-0">
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)',
            backgroundSize: '24px 24px'
          }}
        ></div>
        <div className="absolute bottom-0 left-0 w-full h-64 bg-gradient-to-t from-[#137fec]/10 to-transparent"></div>
      </div>
    </div>
  );
}
