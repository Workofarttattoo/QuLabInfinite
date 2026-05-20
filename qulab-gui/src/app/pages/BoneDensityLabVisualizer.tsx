import { useLabHealth } from '../../lib/hooks';
import { useNavigate } from 'react-router';
import { useState } from 'react';
import { Image3DViewer } from '../components/Image3DViewer';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function BoneDensityLabVisualizer() {
  const { health, loading } = useLabHealth('bone');
  const navigate = useNavigate();
  const [rotation, setRotation] = useState(0);

  return (
    <div className="bg-[#101922] text-white font-['Space_Grotesk'] h-screen w-full overflow-hidden flex flex-col antialiased selection:bg-[#137fec] selection:text-white">
      <style>{`
        ::-webkit-scrollbar {
          width: 4px;
          height: 4px;
        }
        ::-webkit-scrollbar-track {
          background: #101922;
        }
        ::-webkit-scrollbar-thumb {
          background: #334155;
          border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: #137fec;
        }
        .tech-grid {
          background-image:
            linear-gradient(to right, rgba(19, 127, 236, 0.05) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(19, 127, 236, 0.05) 1px, transparent 1px);
          background-size: 40px 40px;
        }
        @keyframes pulse-primary {
          0% { box-shadow: 0 0 0 0 rgba(19, 127, 236, 0.7); }
          70% { box-shadow: 0 0 0 6px rgba(19, 127, 236, 0); }
          100% { box-shadow: 0 0 0 0 rgba(19, 127, 236, 0); }
        }
        .animate-pulse-ring {
          animation: pulse-primary 2s infinite;
        }
        @keyframes scan {
          0% { top: 10%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 90%; opacity: 0; }
        }
      `}</style>

      {/* Background Tech Grid */}
      <div className="fixed inset-0 pointer-events-none tech-grid z-0"></div>

      {/* Top Navigation Bar */}
      <header className="fixed top-0 w-full z-50 glass-panel border-b border-white/5">
        <div className="flex items-center justify-between px-4 py-3 max-w-lg mx-auto w-full">
          <button
            onClick={() => navigate('/labs')}
            className="p-2 -ml-2 rounded-full hover:bg-white/5 active:bg-white/10 transition-colors"
          >
            <span className="material-symbols-outlined text-[#137fec] text-2xl">chevron_left</span>
          </button>
          <div className="flex flex-col items-center">
            <span className="text-xs text-slate-400 uppercase tracking-widest font-bold">Analysis</span>
            <span className="text-sm font-semibold tracking-wide">Bone Density Lab</span>
          </div>
          <div className="flex items-center gap-2 bg-[#137fec]/10 px-3 py-1 rounded-full border border-[#137fec]/20">
            <div className="w-2 h-2 rounded-full bg-[#137fec] animate-pulse-ring"></div>
            <span className="text-[10px] font-bold text-[#137fec] tracking-wider">
              {health ? 'LIVE' : 'OFFLINE'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="flex-1 overflow-y-auto pb-32 pt-20 px-4 max-w-lg mx-auto w-full relative z-10 space-y-4">
        {/* Structural Integrity Visualizer */}
        <section className="relative w-full aspect-[4/5] rounded-xl overflow-hidden border border-white/10 shadow-2xl bg-black/40 group">
          {/* Label Overlay */}
          <div className="absolute top-4 left-4 z-20">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Structural Integrity
            </h2>
            <div className="text-lg font-medium text-white">Femur Cross-Section</div>
          </div>

          {/* 3D Bone Visualization */}
          <div className="absolute inset-0 z-0">
            <Image3DViewer
              imageUrl="https://lh3.googleusercontent.com/aida-public/AB6AXuBY2ICwfFC3vabNk2NxSC_GqsTBqSP3jqMiUG7T34J-cBpUu7Z4w3l5CEi9FHIO0_Dr3ALMrQToQcIJnjY4JfQwewXz834L7Zp_BKuZALzB3R4PkkR-1ebPZG99_umK7zQ9lLFJ-Bfz47JRmzOAYt7ASwes2c6JKsFIoX9D7Bd6AQYt7Qm_aNFt2WhuNvX6XQwpYCNxRSZp6-897tKhAYgCcQNdiAzM9QPJ5CHvfi_XYdXFk3IxPJ6HRa5X_56yQltA2i7mmlCtYj8"
              alt="3D Bone Density Scan"
              className="w-full h-full opacity-60 mix-blend-screen"
              autoRotate={false}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-[#101922] via-transparent to-transparent opacity-90"></div>
            <div className="absolute inset-0 bg-gradient-to-tr from-[#137fec]/20 via-transparent to-purple-500/20 mix-blend-overlay"></div>

            {/* Scanning Line Animation */}
            <div
              className="absolute w-full h-1 bg-[#137fec]/50 shadow-[0_0_15px_rgba(19,127,236,0.8)]"
              style={{ animation: 'scan 3s ease-in-out infinite' }}
            ></div>
          </div>

          {/* Floating Clinical Constants */}
          <div className="absolute top-20 right-4 flex flex-col gap-2 z-20">
            <div className="glass-panel p-2 rounded-lg border border-white/10 backdrop-blur-md">
              <div className="text-[10px] text-slate-400 uppercase">T-Score</div>
              <div className="text-sm font-bold text-white tabular-nums">-0.8</div>
            </div>
            <div className="glass-panel p-2 rounded-lg border border-white/10 backdrop-blur-md">
              <div className="text-[10px] text-slate-400 uppercase">Z-Score</div>
              <div className="text-sm font-bold text-[#137fec] tabular-nums">+0.2</div>
            </div>
            <div className="glass-panel p-2 rounded-lg border border-white/10 backdrop-blur-md">
              <div className="text-[10px] text-slate-400 uppercase">BMD</div>
              <div className="text-sm font-bold text-white tabular-nums">
                0.98 <span className="text-[10px] font-normal text-slate-400">g/cm²</span>
              </div>
            </div>
          </div>

          {/* Density Gradient Legend */}
          <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between z-20">
            <span className="text-[10px] text-slate-400">Low Density</span>
            <div className="flex-1 mx-3 h-1.5 rounded-full bg-gradient-to-r from-blue-900 via-[#137fec] to-white opacity-80 border border-white/20"></div>
            <span className="text-[10px] text-slate-400">High Density</span>
          </div>

          {/* 3D Controls */}
          <div className="absolute right-4 bottom-14 flex flex-col gap-3">
            <button
              onClick={() => setRotation((r) => r + 90)}
              className="w-8 h-8 rounded-full glass-panel flex items-center justify-center text-white hover:bg-[#137fec]/20 transition-colors"
            >
              <span className="material-symbols-outlined text-sm">3d_rotation</span>
            </button>
            <button className="w-8 h-8 rounded-full glass-panel flex items-center justify-center text-white hover:bg-[#137fec]/20 transition-colors">
              <span className="material-symbols-outlined text-sm">zoom_in</span>
            </button>
          </div>
        </section>

        {/* Echo AI Insight Module */}
        <div className="glass-panel rounded-xl p-4 flex items-start gap-3 border-l-2 border-l-[#137fec] relative overflow-hidden">
          <div className="absolute -right-4 -top-4 w-24 h-24 bg-[#137fec]/20 rounded-full blur-2xl"></div>
          <div className="flex-shrink-0 mt-0.5">
            <span className="material-symbols-outlined text-[#137fec] animate-pulse">graphic_eq</span>
          </div>
          <div className="flex-1 relative z-10">
            <div className="flex justify-between items-baseline mb-1">
              <span className="text-xs font-bold text-[#137fec] uppercase tracking-widest">
                Echo AI Insight
              </span>
              <span className="text-[10px] text-slate-500">Just now</span>
            </div>
            <p className="text-sm text-slate-200 leading-relaxed font-light">
              Validating mineral density against the{' '}
              <span className="text-white font-medium">Golden Run</span> benchmark. Variance is within{' '}
              <span className="text-green-400 font-mono">0.002%</span>.
            </p>
          </div>
        </div>

        {/* Stress-Strain Graph Card */}
        <section className="glass-panel rounded-xl p-4 border border-white/5">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold text-white">Young's Modulus</h3>
              <p className="text-[10px] text-slate-400 uppercase tracking-wider">
                Stress (MPa) vs Strain (%)
              </p>
            </div>
            <button className="text-slate-400 hover:text-white">
              <span className="material-symbols-outlined text-sm">more_horiz</span>
            </button>
          </div>

          {/* Graph */}
          <div className="relative h-32 w-full border-l border-b border-slate-700/50">
            <div className="absolute top-0 w-full h-px bg-slate-700/20" style={{ top: '0%' }}></div>
            <div className="absolute top-0 w-full h-px bg-slate-700/20" style={{ top: '25%' }}></div>
            <div className="absolute top-0 w-full h-px bg-slate-700/20" style={{ top: '50%' }}></div>
            <div className="absolute top-0 w-full h-px bg-slate-700/20" style={{ top: '75%' }}></div>

            <svg className="absolute inset-0 w-full h-full overflow-visible" preserveAspectRatio="none">
              <defs>
                <linearGradient id="gradientLine" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#137fec" stopOpacity="0.5"></stop>
                  <stop offset="100%" stopColor="#137fec" stopOpacity="0"></stop>
                </linearGradient>
              </defs>
              <path
                className="opacity-20"
                d="M0,128 L10,120 L30,100 L50,110 L80,60 L120,70 L150,40 L190,50 L220,20 L280,30 L320,5 L320,128 Z"
                fill="url(#gradientLine)"
              ></path>
              <path
                d="M0,128 L10,120 L30,100 L50,110 L80,60 L120,70 L150,40 L190,50 L220,20 L280,30 L320,5"
                fill="none"
                stroke="#137fec"
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              ></path>
              <circle
                className="animate-pulse shadow-[0_0_10px_#137fec]"
                cx="100%"
                cy="5"
                fill="white"
                r="3"
              ></circle>
            </svg>

            <div className="absolute -left-6 top-0 text-[8px] text-slate-500">150</div>
            <div className="absolute -left-6 top-1/2 -translate-y-1/2 text-[8px] text-slate-500">75</div>
            <div className="absolute -left-6 bottom-0 text-[8px] text-slate-500">0</div>
            <div className="absolute left-0 -bottom-4 text-[8px] text-slate-500">0</div>
            <div className="absolute left-1/2 -translate-x-1/2 -bottom-4 text-[8px] text-slate-500">
              2.5
            </div>
            <div className="absolute right-0 -bottom-4 text-[8px] text-slate-500">5.0</div>
          </div>

          <div className="mt-6 flex justify-between items-center text-xs">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-[#137fec]"></div>
              <span className="text-slate-300">Elastic</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-slate-600"></div>
              <span className="text-slate-500">Plastic</span>
            </div>
            <div className="text-[#137fec] font-mono font-bold">142 MPa</div>
          </div>
        </section>
      </main>

      {/* Bottom Control Dock */}
      <nav className="fixed bottom-6 left-4 right-4 z-50">
        <div className="glass-panel rounded-2xl px-6 py-4 flex items-center justify-between shadow-2xl shadow-black/50 border border-white/10 max-w-lg mx-auto">
          <button className="flex flex-col items-center gap-1 group">
            <span className="material-symbols-outlined text-slate-400 group-hover:text-[#137fec] transition-colors text-2xl">
              layers
            </span>
            <span className="text-[9px] font-medium text-slate-500 group-hover:text-[#137fec] uppercase tracking-wider">
              Slice
            </span>
          </button>
          <button className="flex flex-col items-center gap-1 group">
            <span className="material-symbols-outlined text-slate-400 group-hover:text-[#137fec] transition-colors text-2xl">
              rotate_90_degrees_ccw
            </span>
            <span className="text-[9px] font-medium text-slate-500 group-hover:text-[#137fec] uppercase tracking-wider">
              Rotate
            </span>
          </button>

          <button className="relative -top-6 bg-[#137fec] text-white rounded-xl w-14 h-14 flex items-center justify-center shadow-[0_8px_20px_-4px_rgba(19,127,236,0.5)] border border-white/20 hover:scale-105 transition-transform active:scale-95">
            <span className="material-symbols-outlined text-3xl">play_arrow</span>
          </button>

          <button className="flex flex-col items-center gap-1 group">
            <span className="material-symbols-outlined text-slate-400 group-hover:text-[#137fec] transition-colors text-2xl">
              analytics
            </span>
            <span className="text-[9px] font-medium text-slate-500 group-hover:text-[#137fec] uppercase tracking-wider">
              Report
            </span>
          </button>
          <button className="flex flex-col items-center gap-1 group">
            <span className="material-symbols-outlined text-slate-400 group-hover:text-[#137fec] transition-colors text-2xl">
              restart_alt
            </span>
            <span className="text-[9px] font-medium text-slate-500 group-hover:text-[#137fec] uppercase tracking-wider">
              Reset
            </span>
          </button>
        </div>
      </nav>
    </div>
  );
}
