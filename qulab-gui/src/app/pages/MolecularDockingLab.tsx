import { useNavigate } from 'react-router';
import { Image3DViewer } from '../components/Image3DViewer';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function MolecularDockingLab() {
  const navigate = useNavigate();
  const { health, loading: healthLoading } = useLabHealth('drug-discovery');

  return (
    <>
      <style>{`
        body {
          min-height: max(884px, 100dvh);
          background-color: #050505;
          color: #e5e2e1;
          overflow-x: hidden;
        }
        .segmented-progress div {
          height: 4px;
          background: #00dbe9;
          border-right: 2px solid #050505;
        }
        .scanline {
          background: linear-gradient(to bottom, transparent 50%, rgba(0, 219, 233, 0.05) 50.5%, transparent 51%);
          background-size: 100% 4px;
        }
        .crosshair-bg {
          background-image:
            radial-gradient(circle at 50% 50%, transparent 95%, rgba(0, 219, 233, 0.1) 96%),
            linear-gradient(rgba(0, 219, 233, 0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 219, 233, 0.05) 1px, transparent 1px);
          background-size: 100% 100%, 32px 32px, 32px 32px;
        }
      `}</style>

      <div className="bg-[#131313] text-[#e5e2e1] font-['JetBrains_Mono'] selection:bg-[#00f0ff] selection:text-[#006970] overflow-hidden h-screen flex flex-col">
        {/* Top Navigation */}
        <header className="flex justify-between items-center w-full px-4 md:px-8 h-16 bg-[rgba(32,31,31,0.8)] backdrop-blur-md border-b border-[rgba(59,73,75,0.3)] z-50">
          <div className="flex items-center gap-4">
            <div className="w-8 h-8 rounded-full bg-[rgba(0,219,233,0.2)] flex items-center justify-center ring-1 ring-[#00dbe9]">
              <span className="material-symbols-outlined text-[#00dbe9] text-[18px]">biotech</span>
            </div>
            <h1 className="text-[20px] leading-[1.2] font-['Space_Grotesk'] font-bold tracking-tighter text-[#00dbe9]">GLOBAL INTELLIGENCE // ECHO v4.2</h1>
          </div>
          <div className="hidden md:flex items-center gap-8">
            <nav className="flex gap-6">
              <a className="text-[#dbfcff] font-bold text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] ring-1 ring-[#00dbe9] px-3 py-1 cursor-pointer">MOLECULAR DOCKING</a>
              <a className="text-[#b9cacb] hover:bg-white/5 transition-colors text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] px-3 py-1 cursor-pointer">GENOMIC SEQUENCING</a>
              <a className="text-[#b9cacb] hover:bg-white/5 transition-colors text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] px-3 py-1 cursor-pointer">NEURAL MAPPING</a>
            </nav>
            <div className="flex items-center gap-3 border-l border-[rgba(59,73,75,0.3)] pl-8">
              <span className="material-symbols-outlined text-[#00dbe9]" style={{fontVariationSettings: "'FILL' 1"}}>sensors</span>
              <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#00e639]">UNIT 8012 // ONLINE</span>
            </div>
          </div>
        </header>

        {/* Main Workspace */}
              <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="flex-1 flex overflow-hidden p-2 relative crosshair-bg">
          {/* Left Sidebar: Telemetry & Parameters */}
          <aside className="w-80 flex flex-col gap-2 h-full z-10">
            {/* Status Tile */}
            <div className="glass-panel p-4 flex flex-col gap-4 border-l-2 border-l-[#00dbe9]">
              <div className="flex justify-between items-start">
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#849495]">LAB_IDENTIFIER</span>
                <span className="bg-[#00dbe9] text-[#002022] px-2 py-0.5 text-[9px] font-bold">LOCKED</span>
              </div>
              <div>
                <h2 className="text-[20px] leading-[1.2] font-['Space_Grotesk'] text-[#00dbe9] leading-none">MOCKING_UNIT_8012</h2>
                <p className="text-[10px] text-[#849495] mt-1 font-['JetBrains_Mono'] uppercase tracking-tighter">QulabInfinite Stack // Medical Deployment</p>
              </div>
            </div>

            {/* Binding Affinity Tile */}
            <div className="glass-panel p-4 flex flex-col gap-2">
              <div className="flex justify-between items-center">
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#849495]">BINDING_AFFINITY</span>
                <span className="material-symbols-outlined text-[14px] text-[#00e639]">trending_down</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-[24px] leading-[1] tracking-[-0.05em] font-['JetBrains_Mono'] font-medium text-[#00dbe9]">10.42</span>
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#849495]">nM</span>
              </div>
              <div className="mt-2 h-1 w-full bg-[#353534] overflow-hidden flex">
                <div className="w-[72%] bg-[#00dbe9] h-full"></div>
                <div className="w-[1px] bg-[#131313]"></div>
                <div className="w-[10%] bg-[#849495] h-full opacity-50"></div>
              </div>
              <div className="flex justify-between text-[9px] font-['JetBrains_Mono'] text-[#849495] mt-1">
                <span>MIN: 0.12nM</span>
                <span>MAX: 50.00nM</span>
              </div>
            </div>

            {/* Structural Log */}
            <div className="glass-panel flex-1 p-4 flex flex-col gap-3 overflow-hidden">
              <div className="flex justify-between items-center">
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#849495]">ECHO_REASONING_LOG</span>
                <span className="material-symbols-outlined text-[14px] text-[#00dbe9]">terminal</span>
              </div>
              <div className="flex-1 font-['JetBrains_Mono'] text-[11px] leading-relaxed overflow-y-auto space-y-2 pr-2 text-[#b9cacb]">
                <p className="text-[#00e639]">[08:42:11] INITIALIZING DOCKING SEQUENCE...</p>
                <p>[08:42:13] ANALYZING LIGAND CONFORMATIONS: 4,096 DETECTED.</p>
                <p>[08:42:15] EVALUATING SOLVATION ENERGY AT POCKET_A1.</p>
                <p className="text-[#00dbe9]">[08:42:18] HYDROGEN BOND COUNT: 12 (OPTIMAL).</p>
                <p>[08:42:21] DISPERSION INTERACTION DETECTED: -14.2 KCAL/MOL.</p>
                <p>[08:42:25] STOCHASTIC NOISE FILTERING ACTIVE.</p>
                <p className="text-[#c10014]">[08:42:28] STERIC CLASH DETECTED AT RESIDUE 114 (VALINE).</p>
                <p>[08:42:30] RE-ORCHESTRATING ATOM POSITIONING...</p>
                <p className="text-[#00e639]">[08:42:34] SYSTEM STABILIZED. RE-EVALUATING BINDING POSE.</p>
                <p>[08:42:38] DOCKING SCORE IMPROVED BY 12.4%.</p>
                <div className="w-1 h-3 bg-[#00dbe9] animate-pulse inline-block align-middle"></div>
              </div>
            </div>
          </aside>

          {/* Center: Main Visualization */}
          <section className="flex-1 relative mx-2">
            <div className="w-full h-full rounded-lg overflow-hidden glass-panel border-[rgba(0,219,233,0.2)] relative">
              {/* Visual Metadata */}
              <div className="absolute top-6 left-6 z-20 flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-[#00dbe9] rounded-full shadow-[0_0_8px_#00dbe9]"></div>
                  <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#00dbe9]">LIGAND_CORE_VISUALIZATION</span>
                </div>
                <span className="text-[10px] text-[#849495] font-['JetBrains_Mono']">RENDER_MODE: HIGH_FIDELITY_STOCHASTIC</span>
              </div>
              <div className="absolute top-6 right-6 z-20 text-right">
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#00e639]">STABILITY: 98.4%</span>
                <div className="flex gap-0.5 mt-1">
                  <div className="w-3 h-1 bg-[#00e639]"></div>
                  <div className="w-3 h-1 bg-[#00e639]"></div>
                  <div className="w-3 h-1 bg-[#00e639]"></div>
                  <div className="w-3 h-1 bg-[#00e639]"></div>
                  <div className="w-3 h-1 bg-[#353534]"></div>
                </div>
              </div>

              {/* Main Molecular View */}
              <div className="w-full h-full relative overflow-hidden bg-[#050505]">
                <Image3DViewer
                  imageUrl="https://lh3.googleusercontent.com/aida-public/AB6AXuB_DSth0nr32oVT8CpBSn1v7KFempmMl4HIMNvt8co7lRuK5wXa5KVp7rZta5ESdDPVAn8QebFV_rjzAzmPZD8KcKn7TCcg9yIU5ljNeGiq4PaRUZHM-tBHlPUOAFM1Do0Gu4ZjmO5-bGlVH4v8iY86Mg85sEbd5seWI-y_SevI5E2-p4mWCJwZybfWSc4EI9NXuzYrOVJ-QYORzZI042MGg6JiJKEo7inJ-W8Tk4n8awMvAP4XoTDF3ZrE9KOln4hVWl-g3ooHZcs"
                  alt="A high-fidelity 3D microscopic visualization of a molecular docking sequence. Glowing cyan orbs represent atoms bonding with a complex, iridescent violet protein receptor structure in a dark, data-rich environment. The scene is filled with technical overlays, thin stroke UI elements, and a sense of high-tech scientific precision. Lighting is dark and cinematic with neon highlights of cyan and purple against a deep obsidian background, reflecting a functional brutalist and tactical glass OS style."
                  className="w-full h-full"
                  autoRotate={true}
                />
                {/* SVG Overlay Layers */}
                <div className="absolute inset-0 pointer-events-none scanline opacity-30"></div>
              </div>

              {/* Bottom Metrics Bar */}
              <div className="absolute bottom-6 left-6 right-6 z-20 grid grid-cols-4 gap-4">
                <div className="glass-panel p-3 border-t-2 border-t-[#00dbe9]">
                  <span className="text-[9px] text-[#849495] font-['JetBrains_Mono']">SOLVATION_ENERGY</span>
                  <div className="text-lg font-['JetBrains_Mono'] text-[#00dbe9]">-34.12 <span className="text-[10px]">kcal/mol</span></div>
                </div>
                <div className="glass-panel p-3 border-t-2 border-t-[#00e639]">
                  <span className="text-[9px] text-[#849495] font-['JetBrains_Mono']">H_BOND_COUNT</span>
                  <div className="text-lg font-['JetBrains_Mono'] text-[#00e639]">12 <span className="text-[10px]">active</span></div>
                </div>
                <div className="glass-panel p-3 border-t-2 border-t-[#849495]">
                  <span className="text-[9px] text-[#849495] font-['JetBrains_Mono']">RMSD_VALUE</span>
                  <div className="text-lg font-['JetBrains_Mono'] text-[#e5e2e1]">1.84 <span className="text-[10px]">Å</span></div>
                </div>
                <div className="glass-panel p-3 border-t-2 border-t-[#c10014]">
                  <span className="text-[9px] text-[#849495] font-['JetBrains_Mono']">STOCHASTIC_ERROR</span>
                  <div className="text-lg font-['JetBrains_Mono'] text-[#c10014]">0.02%</div>
                </div>
              </div>
            </div>
          </section>

          {/* Right Sidebar: Controls & Actions */}
          <aside className="w-80 flex flex-col gap-2 h-full z-10">
            {/* Command Module */}
            <div className="glass-panel p-4 flex flex-col gap-4">
              <div className="flex justify-between items-center">
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#849495]">COMMAND_MODULE</span>
                <span className="material-symbols-outlined text-[14px] text-[#00dbe9]">settings_input_component</span>
              </div>
              <div className="space-y-3">
                <button className="w-full py-2 border border-[#00dbe9] bg-[rgba(0,219,233,0.1)] text-[#00dbe9] font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] hover:bg-[rgba(0,219,233,0.2)] transition-all active:scale-[0.98] active-stroke flex items-center justify-center gap-2">
                  <span className="material-symbols-outlined text-[16px]">play_arrow</span>
                  RUN_SIMULATION_V4
                </button>
                <button className="w-full py-2 border border-[rgba(59,73,75,1)] text-[#b9cacb] font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] hover:bg-white/5 transition-all flex items-center justify-center gap-2">
                  <span className="material-symbols-outlined text-[16px]">restart_alt</span>
                  RESET_CONFORMATION
                </button>
              </div>
              <div className="pt-2 border-t border-[rgba(59,73,75,0.3)]">
                <span className="text-[10px] text-[#849495] font-['JetBrains_Mono'] mb-2 block uppercase">PARAMETER_FINE_TUNING</span>
                <div className="space-y-4 mt-4">
                  <div className="flex flex-col gap-1">
                    <div className="flex justify-between text-[10px] font-['JetBrains_Mono']">
                      <span>TEMPERATURE_KELVIN</span>
                      <span className="text-[#00dbe9]">310.15K</span>
                    </div>
                    <div className="h-0.5 bg-[#353534] w-full relative">
                      <div className="absolute inset-y-0 left-0 w-[60%] bg-[#00dbe9]"></div>
                      <div className="absolute top-1/2 -translate-y-1/2 left-[60%] w-2 h-2 bg-[#00dbe9] rotate-45"></div>
                    </div>
                  </div>
                  <div className="flex flex-col gap-1">
                    <div className="flex justify-between text-[10px] font-['JetBrains_Mono']">
                      <span>SOLVENT_DIELECTRIC</span>
                      <span className="text-[#00dbe9]">78.4</span>
                    </div>
                    <div className="h-0.5 bg-[#353534] w-full relative">
                      <div className="absolute inset-y-0 left-0 w-[80%] bg-[#00dbe9]"></div>
                      <div className="absolute top-1/2 -translate-y-1/2 left-[80%] w-2 h-2 bg-[#00dbe9] rotate-45"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Target Information */}
            <div className="glass-panel flex-1 p-4 flex flex-col gap-4">
              <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#849495] uppercase">TARGET_RECEPTOR_DATA</span>
              <div className="relative w-full aspect-square bg-[#0e0e0e] rounded border border-[rgba(59,73,75,0.2)] overflow-hidden group">
                <img
                  className="w-full h-full object-cover opacity-60 grayscale group-hover:grayscale-0 transition-all duration-500"
                  alt="A macro scientific photograph of a lab sample with intricate, organic crystalline structures under polarized light. The colors are vibrant teals and deep purples, matching the medical lab interface aesthetic. The image is clean and sharp, emphasizing technical detail and scientific observation, fitting perfectly into a futuristic medical research UI."
                  src="https://lh3.googleusercontent.com/aida-public/AB6AXuAAIGdhNIDxvcikNqUvbcMuAO5bmWMX31KWQZaSQ8q_hPrHbo8TDpa3m-TXUynN7I-486NaarzoTiDwsQ4GVxYQCU3QAchOlpsjIFgQ2THd3GoK51jiOy21CzYf2-Z7CGd7Uai55v95o2xgBBvO3NM3QTOYnhccTML9pQ9KtLCrh6hD6Lbraw53Gdy8YwWF9Cr72cdJTk8psL3QZm0Vu4FXBSIWpjC2fTGnQ-MYC5hH7TD2DNv-W2i2E7auBk_UeIXSCAzq6RSRtWU"
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-12 h-12 border border-[#00dbe9] rounded-full flex items-center justify-center animate-pulse">
                    <span className="material-symbols-outlined text-[#00dbe9]" style={{fontVariationSettings: "'FILL' 1"}}>biotech</span>
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-[10px] text-[#849495] font-['JetBrains_Mono']">ID_KEY</span>
                  <span className="text-[10px] text-[#e5e2e1] font-['JetBrains_Mono']">PDB_ID_7K4S</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[10px] text-[#849495] font-['JetBrains_Mono']">RESOLUTION</span>
                  <span className="text-[10px] text-[#e5e2e1] font-['JetBrains_Mono']">1.45 Å</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[10px] text-[#849495] font-['JetBrains_Mono']">ORGANISM</span>
                  <span className="text-[10px] text-[#e5e2e1] font-['JetBrains_Mono']">H. SAPIENS</span>
                </div>
              </div>
            </div>

            {/* Floating Quick Actions */}
            <div className="flex justify-between gap-2">
              <button className="flex-1 py-3 glass-panel hover:bg-[rgba(0,219,233,0.05)] flex flex-col items-center gap-1 transition-all border-b-2 border-b-transparent hover:border-b-[#00dbe9]">
                <span className="material-symbols-outlined text-[#00dbe9] text-[20px]">save</span>
                <span className="text-[8px] font-['JetBrains_Mono'] text-[#849495]">EXPORT_DATA</span>
              </button>
              <button className="flex-1 py-3 glass-panel hover:bg-[rgba(0,230,57,0.05)] flex flex-col items-center gap-1 transition-all border-b-2 border-b-transparent hover:border-b-[#00e639]">
                <span className="material-symbols-outlined text-[#00e639] text-[20px]">share</span>
                <span className="text-[8px] font-['JetBrains_Mono'] text-[#849495]">TRANSMIT</span>
              </button>
              <button className="flex-1 py-3 glass-panel hover:bg-[rgba(193,0,20,0.05)] flex flex-col items-center gap-1 transition-all border-b-2 border-b-transparent hover:border-b-[#c10014]">
                <span className="material-symbols-outlined text-[#c10014] text-[20px]">emergency</span>
                <span className="text-[8px] font-['JetBrains_Mono'] text-[#849495]">ABORT</span>
              </button>
            </div>
          </aside>
        </main>

        {/* Bottom Navigation Bar */}
        <nav className="fixed bottom-0 left-0 w-full z-50 flex justify-around items-center px-4 h-20 bg-[rgba(28,27,27,0.9)] backdrop-blur-xl border-t border-[rgba(59,73,75,0.2)]">
          <button className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200">
            <span className="material-symbols-outlined">grid_view</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] mt-1">TELEMETRY</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200">
            <span className="material-symbols-outlined">hub</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] mt-1">ORCHESTRATION</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#00dbe9] border-t-2 border-[#00dbe9] pt-2 scale-95 opacity-80">
            <span className="material-symbols-outlined" style={{fontVariationSettings: "'FILL' 1"}}>science</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] mt-1">LABS</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200">
            <span className="material-symbols-outlined">monitoring</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] mt-1">RESEARCH</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200">
            <span className="material-symbols-outlined">terminal</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] mt-1">COMMAND</span>
          </button>
        </nav>

        {/* UI Accents */}
        <div className="fixed bottom-24 right-8 z-50 flex flex-col items-end gap-2 pointer-events-none">
          <div className="text-[8px] font-['JetBrains_Mono'] text-[#00dbe9] bg-[rgba(0,219,233,0.1)] px-2 py-0.5 border border-[rgba(0,219,233,0.2)]">LATENCY: 12ms</div>
          <div className="text-[8px] font-['JetBrains_Mono'] text-[#00e639] bg-[rgba(0,230,57,0.1)] px-2 py-0.5 border border-[rgba(0,230,57,0.2)]">UPTIME: 99.9%</div>
        </div>
      </div>
    </>
  );
}
