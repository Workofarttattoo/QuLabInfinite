import { useLabHealth } from '../../lib/hooks';
import { useNavigate } from 'react-router';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function NeuroProductionVisualizer() {
  const { health, loading } = useLabHealth('neuro');
  const navigate = useNavigate();

  return (
    <div className="bg-[#080c11] font-['Space_Grotesk'] text-white min-h-screen flex flex-col overflow-hidden relative selection:bg-[#137fec] selection:text-white">
      <style>{`
        .glass-high-fi {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(32px);
          -webkit-backdrop-filter: blur(32px);
          border: 1px solid rgba(255, 255, 255, 0.12);
          box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        }

        .glass-inner-depth {
          position: relative;
          overflow: hidden;
          background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 100%);
        }

        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .no-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }

        .neon-glow {
          box-shadow: 0 0 15px rgba(19, 127, 236, 0.4);
        }

        .waveform-bar {
          width: 2px;
          background: #137fec;
          border-radius: 9999px;
        }
      `}</style>

      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-10%] right-[-10%] w-[600px] h-[600px] bg-[#137fec]/10 rounded-full blur-[120px]"></div>
        <div className="absolute bottom-[-10%] left-[-20%] w-[500px] h-[500px] bg-indigo-950/30 rounded-full blur-[100px]"></div>
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-30 mix-blend-overlay"></div>
      </div>

      <aside className="fixed right-0 top-1/4 z-50 flex flex-col gap-2 pointer-events-none">
        <div className="glass-high-fi p-3 rounded-l-2xl border-r-0 pointer-events-auto flex flex-col items-center gap-4 py-6 shadow-2xl">
          <span className="material-symbols-outlined text-[#137fec] text-xl">tune</span>
          <div className="h-px w-6 bg-white/10"></div>
          <div className="[writing-mode:vertical-lr] text-[10px] font-bold tracking-tighter text-white/40 uppercase rotate-180">Clinical Constants</div>
        </div>
      </aside>

      <header className="relative z-20 flex items-center justify-between px-6 pt-14 pb-4 w-full">
        <button
          onClick={() => navigate('/labs')}
          className="w-10 h-10 flex items-center justify-center rounded-xl glass-high-fi"
        >
          <span className="material-symbols-outlined text-white text-xl">grid_view</span>
        </button>
        <div className="flex flex-col items-center">
          <span className="text-[10px] font-black tracking-[0.3em] text-[#137fec] uppercase">Qulab-8001</span>
          <span className="text-sm font-semibold text-white/95">Neuro-Production Suite</span>
        </div>
        <div className="w-10 h-10 flex items-center justify-center rounded-xl glass-high-fi relative">
          <span className="material-symbols-outlined text-[#137fec] text-xl">analytics</span>
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full border-2 border-[#080c11]"></div>
        </div>
      </header>

            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="relative z-10 flex-1 flex flex-col px-6 gap-5 overflow-y-auto no-scrollbar pb-32">
        <section className="relative w-full aspect-square mt-2 flex items-center justify-center">
          <div className="absolute inset-0 border border-white/5 rounded-full scale-100"></div>
          <div className="absolute inset-0 border border-[#137fec]/10 rounded-full scale-[0.85] border-dashed"></div>
          <div className="relative w-[85%] h-[85%] glass-inner-depth rounded-full flex items-center justify-center shadow-inner">
            <div className="absolute inset-0 bg-gradient-to-tr from-[#137fec]/5 via-transparent to-red-500/5 opacity-40"></div>
            <img
              alt="Neural Map"
              className="w-4/5 h-4/5 object-contain filter drop-shadow-[0_0_30px_rgba(19,127,236,0.5)] contrast-110"
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuCtOL1H_iWnsHLr81tTXy0PPIbhCtXY3s1dQwWgjlbLx3T-qj_PD35SMfUgi4Zw5FffhPtEn93XDw8ghlt85b_Etps_VottcR9fFf21CL5wWE_cwFSSheSwcr17eZxICOxPrgGlj1oxEAUNjmEVazLIDfRspiNcCUyKUw0SWwPh4GCJYIpgUtPlDCYYPuGuur-Kh-E6igeTlcMyz4JATOaHjEEdz22L0pHOZPfBwUVIRbBJqwsgekM-NK0KtmEGkw5Ok62fLIbHbw4"
            />
            <div className="absolute top-[35%] left-[45%] w-16 h-16 bg-red-500/10 rounded-full blur-xl border border-red-500/20"></div>
            <div className="absolute bottom-[25%] right-[30%] w-12 h-12 bg-yellow-400/10 rounded-full blur-xl border border-yellow-400/20"></div>
            <div className="absolute top-1/4 left-1/3 w-1 h-1 bg-[#00f2ff] rounded-full shadow-[0_0_8px_#00f2ff]"></div>
            <div className="absolute bottom-1/3 right-1/4 w-1 h-1 bg-[#00f2ff] rounded-full shadow-[0_0_8px_#00f2ff]"></div>
          </div>
          <div className="absolute top-0 left-0 glass-high-fi px-3 py-1.5 rounded-lg flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-[#00f2ff] neon-glow"></div>
            <span className="text-[9px] font-bold tracking-widest text-white/80 uppercase">NIST: Traceable</span>
          </div>
        </section>

        <div className="flex justify-between items-center glass-high-fi p-1.5 rounded-2xl">
          <button className="flex-1 bg-[#137fec]/20 border border-[#137fec]/30 py-2.5 rounded-xl text-[10px] font-black text-white uppercase tracking-widest">Synaptic</button>
          <button className="flex-1 py-2.5 text-[10px] font-bold text-white/40 uppercase tracking-widest">Pathology</button>
          <button className="flex-1 py-2.5 text-[10px] font-bold text-white/40 uppercase tracking-widest">Thermal</button>
        </div>

        <div className="glass-high-fi p-5 rounded-3xl flex items-center gap-5 relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-32 h-32 bg-[#137fec]/5 rounded-full -mr-16 -mt-16 blur-3xl"></div>
          <div className="relative shrink-0">
            <div className="w-20 h-20 rounded-2xl overflow-hidden border border-white/20 p-1 relative shadow-2xl">
              <img
                alt="Echo HD"
                className="w-full h-full object-cover rounded-xl filter contrast-125 saturate-150"
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuCBotyawt6tvF5JXJnPuUnEKbvPIBDXwr_3tSLN17lKqotTcN_JseeytROSzZYyRGgmN8uYfsypIbJuD3G3vRXr2u_1K--nHUvcYjSWPgFpzaOTo8nZlq2JvyfGSEqmIIgvAnLI0Ys-B5wrXBDHg861uh_ekP-2ld4JLBFHtwN4snrDT3uYiTiAzS07i-aHQq7WwJYW1psz641Du4Mk10omrf0fcIv-40-dBvGLIHMKL3lQw1H6rdpy2Y-gkZzVlsw8J1rQ1bzJlls"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-[#080c11]/80 to-transparent"></div>
            </div>
            <div className="absolute -bottom-2 -right-2 bg-[#137fec] rounded-lg p-1 shadow-lg">
              <span className="material-symbols-outlined text-[14px] text-white">electric_bolt</span>
            </div>
          </div>
          <div className="flex-1 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-black text-[#137fec] tracking-widest uppercase">Presence: Echo</span>
              <div className="flex gap-0.5 items-end h-4">
                <div className="waveform-bar h-2"></div>
                <div className="waveform-bar h-4"></div>
                <div className="waveform-bar h-3"></div>
                <div className="waveform-bar h-5"></div>
                <div className="waveform-bar h-2"></div>
                <div className="waveform-bar h-3"></div>
                <div className="waveform-bar h-1"></div>
              </div>
            </div>
            <p className="text-xs leading-relaxed text-gray-100 font-medium">
              "Structural binding affinity is normalizing at 4.2nM. Comparing current slice against 'Golden Run' benchmarks now."
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="glass-high-fi p-4 rounded-2xl flex flex-col gap-1">
            <span className="text-[9px] font-bold text-white/40 uppercase tracking-widest">Protein Conc.</span>
            <div className="flex items-baseline gap-1">
              <span className="text-lg font-mono font-bold text-white">{health?.proteinConc || '24.8'}</span>
              <span className="text-[10px] text-[#137fec] font-bold">mg/mL</span>
            </div>
            <div className="w-full h-1 bg-white/5 rounded-full mt-2">
              <div className="w-3/4 h-full bg-[#137fec] rounded-full"></div>
            </div>
          </div>
          <div className="glass-high-fi p-4 rounded-2xl flex flex-col gap-1">
            <span className="text-[9px] font-bold text-white/40 uppercase tracking-widest">Binding Affinity</span>
            <div className="flex items-baseline gap-1">
              <span className="text-lg font-mono font-bold text-white">4.2</span>
              <span className="text-[10px] text-[#00f2ff] font-bold">nM</span>
            </div>
            <div className="w-full h-1 bg-white/5 rounded-full mt-2">
              <div className="w-1/2 h-full bg-[#00f2ff] rounded-full"></div>
            </div>
          </div>
        </div>
      </main>

      <footer className="fixed bottom-0 left-0 right-0 z-40">
        <div className="mx-4 mb-8 glass-high-fi rounded-2xl overflow-hidden p-4 flex flex-col gap-3 shadow-[0_-10px_40px_rgba(0,0,0,0.5)] border-t border-white/20">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2">
              <span className="material-symbols-outlined text-[#137fec] text-sm">show_chart</span>
              <h3 className="text-[10px] font-black text-white/80 uppercase tracking-[0.2em]">Golden Run Comparison</h3>
            </div>
            <div className="flex gap-4">
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-[#137fec]"></div>
                <span className="text-[8px] font-bold text-white/40 uppercase">Live</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-white/20"></div>
                <span className="text-[8px] font-bold text-white/40 uppercase">Benchmark</span>
              </div>
            </div>
          </div>
          <div className="h-16 w-full flex items-end gap-1 px-1">
            {[60, 75, 90, 80, 95, 70, 65, 85].map((height, i) => (
              <div key={i} className="flex-1 bg-white/5 rounded-t-sm relative" style={{ height: `${height}%` }}>
                <div className="absolute inset-x-0 bottom-0 bg-[#137fec]/40 border-t border-[#137fec]" style={{ height: `${height}%` }}></div>
                {i === 4 && <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-1 h-1 bg-white rounded-full"></div>}
              </div>
            ))}
          </div>
          <div className="flex justify-between items-center text-[9px] font-mono text-white/40 border-t border-white/5 pt-2">
            <span>LATENCY: 0.00ms</span>
            <span className="text-[#137fec] font-bold">MATCH: {health?.match || '99.42%'}</span>
          </div>
        </div>
        <div className="absolute bottom-1.5 left-1/2 -translate-x-1/2 w-32 h-1 bg-white/20 rounded-full"></div>
      </footer>
    </div>
  );
}
