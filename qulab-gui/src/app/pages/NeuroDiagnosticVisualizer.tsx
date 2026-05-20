import { useLabHealth } from '../../lib/hooks';
import { useNavigate } from 'react-router';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function NeuroDiagnosticVisualizer() {
  const { health, loading } = useLabHealth('neuro');
  const navigate = useNavigate();

  return (
    <div className="bg-[#101922] font-['Space_Grotesk'] text-white min-h-screen flex flex-col overflow-hidden relative selection:bg-[#137fec] selection:text-white">
      <style>{`
        .no-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .no-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }

        .neon-text {
          text-shadow: 0 0 10px rgba(19, 127, 236, 0.5);
        }

        @keyframes pulse-slow {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .animate-pulse-slow {
          animation: pulse-slow 3s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }

        .animate-spin-slow {
          animation: spin-slow 12s linear infinite;
        }
      `}</style>

      {/* Background Ambience */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute top-[-10%] right-[-10%] w-[500px] h-[500px] bg-[#137fec]/20 rounded-full blur-[100px] animate-pulse-slow"></div>
        <div className="absolute bottom-[-10%] left-[-20%] w-[400px] h-[400px] bg-indigo-900/40 rounded-full blur-[80px]"></div>
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay"></div>
      </div>

      {/* Header */}
      <header className="relative z-20 flex items-center justify-between px-6 pt-12 pb-4 w-full">
        <button
          onClick={() => navigate('/labs')}
          className="p-2 rounded-full glass-panel hover:bg-white/10 transition-colors"
        >
          <span className="material-symbols-outlined text-white text-xl">arrow_back</span>
        </button>
        <div className="flex flex-col items-center">
          <span className="text-xs font-bold tracking-widest text-[#137fec]/80 uppercase">Qulab-8001</span>
          <span className="text-sm font-medium text-white/90">Alzheimer's Neuro-Diagnostic</span>
        </div>
        <div className="p-2 rounded-full glass-panel">
          <div className="w-5 h-5 relative flex items-center justify-center">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#137fec] opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-[#137fec]"></span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="relative z-10 flex-1 flex flex-col px-6 gap-6 overflow-y-auto no-scrollbar pb-24">
        {/* 3D Neural Visualizer */}
        <section className="relative w-full aspect-square mt-2">
          <div className="absolute inset-0 bg-[#137fec]/5 rounded-full blur-3xl transform scale-75"></div>
          <div className="relative w-full h-full flex items-center justify-center">
            <div className="absolute w-[90%] h-[90%] border border-[#137fec]/20 rounded-full animate-spin-slow border-dashed"></div>
            <div className="absolute w-[95%] h-[95%] border border-white/5 rounded-full"></div>
            <div className="relative w-[70%] h-[70%] z-10">
              <img
                alt="Neural Map"
                className="w-full h-full object-contain drop-shadow-[0_0_25px_rgba(19,127,236,0.6)] animate-pulse-slow opacity-90"
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuCtOL1H_iWnsHLr81tTXy0PPIbhCtXY3s1dQwWgjlbLx3T-qj_PD35SMfUgi4Zw5FffhPtEn93XDw8ghlt85b_Etps_VottcR9fFf21CL5wWE_cwFSSheSwcr17eZxICOxPrgGlj1oxEAUNjmEVazLIDfRspiNcCUyKUw0SWwPh4GCJYIpgUtPlDCYYPuGuur-Kh-E6igeTlcMyz4JATOaHjEEdz22L0pHOZPfBwUVIRbBJqwsgekM-NK0KtmEGkw5Ok62fLIbHbw4"
              />
              <div className="absolute top-[20%] right-[20%] w-3 h-3 bg-white rounded-full shadow-[0_0_10px_white] animate-ping"></div>
              <div className="absolute bottom-[30%] left-[25%] w-2 h-2 bg-[#137fec] rounded-full shadow-[0_0_10px_#137fec]"></div>
            </div>
            <div className="absolute top-4 left-0 glass-panel px-3 py-1 rounded-full flex items-center gap-2 scale-75 origin-left">
              <span className="w-1.5 h-1.5 bg-green-400 rounded-full"></span>
              <span className="text-[10px] tracking-wider uppercase text-white/70">Synapse: OK</span>
            </div>
            <div className="absolute bottom-10 right-0 glass-panel px-3 py-1 rounded-full flex items-center gap-2 scale-75 origin-right">
              <span className="w-1.5 h-1.5 bg-[#137fec] rounded-full"></span>
              <span className="text-[10px] tracking-wider uppercase text-white/70">Cortex Scan</span>
            </div>
          </div>
        </section>

        {/* View Controls */}
        <div className="flex justify-center gap-3">
          <button className="glass-panel-active px-5 py-2 rounded-full text-xs font-bold text-white uppercase tracking-wider shadow-[0_0_15px_rgba(19,127,236,0.2)]">
            Synaptic
          </button>
          <button className="glass-panel px-5 py-2 rounded-full text-xs font-bold text-white/60 hover:text-white uppercase tracking-wider transition-colors">
            Thermal
          </button>
          <button className="glass-panel px-5 py-2 rounded-full text-xs font-bold text-white/60 hover:text-white uppercase tracking-wider transition-colors">
            Structural
          </button>
        </div>

        {/* Echo Persona Widget */}
        <div className="glass-panel p-4 rounded-xl flex items-start gap-4 transform transition-all hover:scale-[1.02] duration-300">
          <div className="relative shrink-0">
            <div className="w-14 h-14 rounded-full overflow-hidden border-2 border-[#137fec] p-0.5 relative">
              <img
                alt="Echo Avatar"
                className="w-full h-full object-cover rounded-full filter grayscale contrast-125"
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuCBotyawt6tvF5JXJnPuUnEKbvPIBDXwr_3tSLN17lKqotTcN_JseeytROSzZYyRGgmN8uYfsypIbJuD3G3vRXr2u_1K--nHUvcYjSWPgFpzaOTo8nZlq2JvyfGSEqmIIgvAnLI0Ys-B5wrXBDHg861uh_ekP-2ld4JLBFHtwN4snrDT3uYiTiAzS07i-aHQq7WwJYW1psz641Du4Mk10omrf0fcIv-40-dBvGLIHMKL3lQw1H6rdpy2Y-gkZzVlsw8J1rQ1bzJlls"
              />
              <div className="absolute inset-0 bg-[#137fec]/20 mix-blend-overlay"></div>
            </div>
            <div className="absolute -bottom-1 -right-1 bg-[#101922] rounded-full p-1">
              <span className="material-symbols-outlined text-[#137fec] text-sm">mic</span>
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-[#137fec] tracking-wider uppercase">Echo AI // Assistant</span>
            </div>
            <p className="text-sm leading-relaxed text-gray-200">
              "Monitoring amyloid-beta aggregation in real-time. Precision maintained at NIST standards."
            </p>
          </div>
        </div>

        {/* Protein Folding Feed */}
        <div className="glass-panel p-4 rounded-xl flex flex-col gap-3">
          <div className="flex justify-between items-center border-b border-white/10 pb-2">
            <h3 className="text-xs font-bold text-white/80 uppercase tracking-widest flex items-center gap-2">
              <span className="material-symbols-outlined text-[#137fec] text-sm">science</span>
              Folding Simulation
            </h3>
            <span className="text-[10px] text-[#137fec]/80 font-mono">BETA-SHEET FORMATION</span>
          </div>
          <div className="flex gap-4 items-center">
            <div className="w-24 h-24 rounded-lg bg-black/40 relative overflow-hidden border border-white/5 shrink-0">
              <img
                alt="Protein Sim"
                className="w-full h-full object-cover opacity-80"
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuAJ0pvhu823UoJax0HFkrTxBTGLILuaTFHOzIP-lsnVrJQXEZh1DFZBrGaTVyEPlPMsm_LPq7FqupA2wOZ5s_EIUqLCs4NCrwOU6IiJNAkFNqAUUzyKHGsb2L4UTZ8BTzh0YPQga63cDDAFN_tfcy2LlIPYop8CScPaD0zAJZyuPuZRkLFky2OcPEGtx7SWVn3gFx7tnENgcvB4J5BbTy4sQq9x1eorFy5S1xdmqsHzjEMt_S6eHj3U5TLMBav7ig_TpW27pgIJSDE"
              />
              <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTEgMWgydjJIMUMxeiIgZmlsbD0iI2ZmZiIgZmlsbC1vcGFjaXR5PSIwLjEiLz48L3N2Zz4=')] opacity-30"></div>
            </div>
            <div className="flex-1 grid grid-cols-2 gap-y-3 gap-x-2">
              <div>
                <div className="text-[10px] text-gray-400 uppercase">Stability</div>
                <div className="text-sm font-mono text-white">{health?.stability || '98.4%'}</div>
              </div>
              <div>
                <div className="text-[10px] text-gray-400 uppercase">Temp</div>
                <div className="text-sm font-mono text-white">37.2°C</div>
              </div>
              <div>
                <div className="text-[10px] text-gray-400 uppercase">Rate</div>
                <div className="text-sm font-mono text-[#137fec] neon-text">45ms</div>
              </div>
              <div>
                <div className="text-[10px] text-gray-400 uppercase">Stage</div>
                <div className="text-sm font-mono text-white">IV</div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Technical Log Terminal */}
      <footer className="fixed bottom-0 left-0 right-0 z-30">
        <div className="mx-4 mb-4 glass-panel rounded-xl overflow-hidden shadow-2xl backdrop-blur-xl bg-black/60 border-t border-[#137fec]/20">
          <div className="bg-black/40 px-3 py-1.5 flex items-center justify-between border-b border-white/5">
            <span className="text-[10px] text-gray-500 font-mono uppercase">System_Logs_v4.2</span>
            <div className="flex gap-1.5">
              <div className="w-2 h-2 rounded-full bg-red-500/50"></div>
              <div className="w-2 h-2 rounded-full bg-yellow-500/50"></div>
              <div className="w-2 h-2 rounded-full bg-green-500/50"></div>
            </div>
          </div>
          <div className="p-3 font-mono text-[10px] h-24 overflow-hidden relative">
            <div className="absolute top-0 left-0 right-0 h-4 bg-gradient-to-b from-black/20 to-transparent pointer-events-none"></div>
            <div className="absolute bottom-0 left-0 right-0 h-8 bg-gradient-to-t from-[#182430] to-transparent pointer-events-none"></div>
            <ul className="space-y-1 text-gray-300">
              <li className="opacity-40"><span className="text-[#137fec]">[14:02:44]</span> INIT_SEQ_START // Qulab_Core</li>
              <li className="opacity-60"><span className="text-[#137fec]">[14:02:45]</span> SYNC_PACKET_RECEIVED // Node_81</li>
              <li className="opacity-80"><span className="text-[#137fec]">[14:02:46]</span> NIST_CALIB_OK // Variance: 0.001</li>
              <li><span className="text-[#137fec]">[14:02:47]</span> THRESHOLD_CHECK... <span className="text-green-400">PASS</span></li>
              <li><span className="text-[#137fec]">[14:02:48]</span> AMYLOID_DETECTED // Sector 4B</li>
              <li className="animate-pulse"><span className="text-[#137fec]">[14:02:49]</span> {health ? 'PROCESSING_LIVE_FEED...' : 'OFFLINE...'}</li>
            </ul>
          </div>
        </div>
      </footer>
    </div>
  );
}
