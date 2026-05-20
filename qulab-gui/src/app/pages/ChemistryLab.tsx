import { Navigation } from '../components/Navigation';
import { useLabHealth, useLabThresholds } from '../../lib/hooks';
import { useState } from 'react';
import { Image3DViewer } from '../components/Image3DViewer';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function ChemistryLab() {
  const { health, loading: healthLoading } = useLabHealth('chemistry');
  const [activeView, setActiveView] = useState<'quantum' | 'molecular' | 'reactivity'>('quantum');

  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 px-[32px]">
          <EchoLabCommandInline className="mb-8" />
        <div className="max-w-[1440px] mx-auto">
          <div className="mb-12">
            <div className="mb-4 inline-flex items-center gap-2 px-3 py-1 bg-[rgba(0,240,255,0.2)] border border-[rgba(0,240,255,0.4)] rounded-full">
              <span className="w-2 h-2 rounded-full bg-[#00dbe9] animate-pulse"></span>
              <span className="text-[12px] leading-[16px] tracking-[0.15em] font-bold text-[#00dbe9]">QUANTUM_CHEMISTRY_SIMULATOR</span>
            </div>
            <h1 className="text-[48px] leading-[56px] tracking-[-0.02em] font-bold text-[#00dbe9] mb-4">
              Quantum Chemistry Simulator
            </h1>
            <p className="text-[18px] leading-[28px] text-[#b9cacb] max-w-3xl">
              Ab initio quantum mechanical calculations - Electronic structure, molecular orbital theory, and reaction pathway analysis - Port 8012
            </p>
          </div>

          <div className="grid grid-cols-4 gap-6 mb-8">
            <div className="glass-panel p-6 rounded-xl neon-glow-cyan col-span-1">
              <h3 className="text-[24px] font-semibold text-[#00dbe9] mb-4">Lab Status</h3>
              {healthLoading ? (
                <div className="text-[#b9cacb]">Loading...</div>
              ) : health ? (
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-3 h-3 rounded-full bg-[#00dbe9] animate-pulse"></span>
                    <span className="text-[#00dbe9] font-semibold">✅ {health.status}</span>
                  </div>
                  <div className="text-[12px] text-[#b9cacb] mt-4">Quantum Engine Online</div>
                </div>
              ) : (
                <div className="text-[#ffb4ab]">❌ Offline</div>
              )}
            </div>

            <div className="glass-panel p-6 rounded-xl border-white/10">
              <h3 className="text-[16px] font-semibold text-[#ddb7ff] mb-4">Theory Levels</h3>
              <div className="space-y-2 text-[11px]">
                <div className="px-2 py-1 bg-[rgba(0,219,233,0.15)] rounded text-[#00dbe9]">HF (Hartree-Fock)</div>
                <div className="px-2 py-1 bg-[rgba(0,219,233,0.15)] rounded text-[#00dbe9]">MP2/MP4</div>
                <div className="px-2 py-1 bg-[rgba(0,219,233,0.15)] rounded text-[#00dbe9]">CCSD(T)</div>
              </div>
            </div>

            <div className="glass-panel p-6 rounded-xl border-white/10">
              <h3 className="text-[16px] font-semibold text-[#ddb7ff] mb-4">Basis Sets</h3>
              <div className="space-y-2 text-[11px]">
                <div className="px-2 py-1 bg-[rgba(168,85,247,0.15)] rounded text-[#ddb7ff]">STO-3G</div>
                <div className="px-2 py-1 bg-[rgba(168,85,247,0.15)] rounded text-[#ddb7ff]">6-31G*</div>
                <div className="px-2 py-1 bg-[rgba(168,85,247,0.15)] rounded text-[#ddb7ff]">cc-pVTZ</div>
              </div>
            </div>

            <div className="glass-panel p-6 rounded-xl border-white/10">
              <h3 className="text-[16px] font-semibold text-[#00dbe9] mb-4">DFT Functionals</h3>
              <div className="space-y-2 text-[11px]">
                <div className="px-2 py-1 bg-[rgba(0,219,233,0.15)] rounded text-[#00dbe9]">B3LYP</div>
                <div className="px-2 py-1 bg-[rgba(0,219,233,0.15)] rounded text-[#00dbe9]">PBE0</div>
                <div className="px-2 py-1 bg-[rgba(0,219,233,0.15)] rounded text-[#00dbe9]">ωB97X-D</div>
              </div>
            </div>
          </div>

          {/* Crystal Synthesis Visualization */}
          <div className="grid grid-cols-2 gap-6 mb-8">
            <div className="glass-panel p-6 rounded-xl border-[#00dbe9]/20 relative overflow-hidden">
              <div className="absolute top-4 left-4 z-10">
                <h3 className="text-[20px] font-semibold text-[#00dbe9] mb-1">Crystal Growth Simulation</h3>
                <p className="text-[12px] text-[#b9cacb]">Sodium Chloride Lattice Formation</p>
              </div>

              {/* 3D Crystal Visualization */}
              <div className="relative w-full aspect-square flex items-center justify-center mt-12">
                <Image3DViewer
                  imageUrl="https://lh3.googleusercontent.com/aida-public/AB6AXuArJd2RXw-pRHr1jsHcF2jPYPIUV-0K4W-5v1lJyit10TCl9qiIkaDmv7Y9_kM0z_ok3u8OMgk-PdwPoWHG9UlsZNPMzhexVD7jshXGaIInChpxMCKvY1LTv6F1iAWNi9O11M2hp34eISOBWYLk9yVtxUyqViFYnBEgqqcrQ5ekVhjAzPosyDOtJKgOLE9OpEW2R503QJPb4jpBmUpWL_fENwNWNQq7lhsLV52esyMn5wgRJTxOHq2utMPS65-17oe7vBupmSp7jVg"
                  alt="NaCl Crystal Lattice 3D"
                  className="w-full h-full opacity-90"
                  autoRotate={true}
                />
                {/* Corner brackets */}
                <div className="absolute top-2 left-2 w-8 h-8 border-t-2 border-l-2 border-[#00dbe9]/50 pointer-events-none"></div>
                <div className="absolute bottom-2 right-2 w-8 h-8 border-b-2 border-r-2 border-[#00dbe9]/50 pointer-events-none"></div>
              </div>

              <div className="mt-4 grid grid-cols-3 gap-2">
                <div className="text-center p-2 bg-[#00dbe9]/10 rounded">
                  <div className="text-[10px] text-[#b9cacb]">UNIT CELLS</div>
                  <div className="text-[14px] font-bold text-[#00dbe9]">8³</div>
                </div>
                <div className="text-center p-2 bg-[#ddb7ff]/10 rounded">
                  <div className="text-[10px] text-[#b9cacb]">TEMP</div>
                  <div className="text-[14px] font-bold text-[#ddb7ff]">298K</div>
                </div>
                <div className="text-center p-2 bg-green-500/10 rounded">
                  <div className="text-[10px] text-[#b9cacb]">PURITY</div>
                  <div className="text-[14px] font-bold text-green-400">99.8%</div>
                </div>
              </div>
            </div>

            {/* Echo AI Crystal Growth Guide */}
            <div className="glass-panel p-6 rounded-xl border-[#ddb7ff]/20 relative overflow-hidden">
              <div className="absolute -right-8 -top-8 w-32 h-32 bg-[#ddb7ff]/10 rounded-full blur-2xl"></div>

              <div className="flex items-start gap-3 mb-4">
                <div className="w-10 h-10 rounded-full bg-[#ddb7ff]/20 flex items-center justify-center">
                  <span className="material-symbols-outlined text-[#ddb7ff] animate-pulse">science</span>
                </div>
                <div>
                  <h4 className="text-[14px] font-bold text-[#ddb7ff] uppercase tracking-wider">Echo AI Lab Guide</h4>
                  <p className="text-[10px] text-[#b9cacb]">Crystal Synthesis Protocol</p>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <h5 className="text-[12px] font-semibold text-white mb-2">Step-by-Step Crystal Growth:</h5>
                  <ol className="text-[12px] text-[#b9cacb] leading-relaxed space-y-2">
                    <li className="flex gap-2">
                      <span className="text-[#00dbe9] font-bold">1.</span>
                      <span>Dissolve 36g NaCl in 100mL H₂O at <span className="text-[#00dbe9] font-medium">60°C</span></span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-[#00dbe9] font-bold">2.</span>
                      <span>Cool slowly to room temp (<span className="text-yellow-400 font-medium">1°C per hour</span>)</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-[#00dbe9] font-bold">3.</span>
                      <span>Observe nucleation starting after <span className="text-[#ddb7ff] font-medium">~4 hours</span></span>
                    </li>
                  </ol>
                </div>

                {/* Visual Quality Check */}
                <div className="bg-[#0a0e14]/80 p-3 rounded-lg border border-white/5">
                  <div className="text-[10px] font-bold text-[#00dbe9] uppercase mb-2">Crystal Quality Check:</div>
                  <div className="grid grid-cols-2 gap-2 mb-2">
                    <div className="h-16 bg-gradient-to-br from-white/90 to-white/60 rounded flex flex-col items-center justify-center border border-green-400">
                      <span className="text-[10px] font-bold text-green-600">✓ Clear & Cubic</span>
                      <span className="text-[8px] text-slate-600">Perfect lattice</span>
                    </div>
                    <div className="h-16 bg-gradient-to-br from-gray-400 to-gray-600 rounded flex flex-col items-center justify-center border border-red-400">
                      <span className="text-[10px] font-bold text-red-400">✗ Cloudy</span>
                      <span className="text-[8px] text-white">Impurities</span>
                    </div>
                  </div>
                  <p className="text-[10px] text-[#b9cacb]">
                    Perfect crystals are <span className="text-white">clear and cubic</span> with sharp edges.
                    Cloudy crystals indicate <span className="text-red-400">contamination</span> - recrystallize from purer solvent.
                  </p>
                </div>

                <div className="flex gap-2">
                  <button className="flex-1 px-3 py-2 bg-[#ddb7ff]/20 hover:bg-[#ddb7ff]/30 rounded text-[11px] font-semibold text-[#ddb7ff] transition-all">
                    View Protocol →
                  </button>
                  <button className="px-3 py-2 bg-white/5 hover:bg-white/10 rounded text-[11px] text-[#b9cacb] transition-all">
                    ⚗️ Lab Notes
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-panel p-8 rounded-xl mb-8">
            <div className="flex gap-4 mb-6 border-b border-white/10 pb-4">
              <button
                onClick={() => setActiveView('quantum')}
                className={`px-6 py-2 text-[12px] tracking-[0.15em] font-bold uppercase transition-all ${
                  activeView === 'quantum'
                    ? 'text-[#00dbe9] bg-[rgba(0,219,233,0.2)] rounded'
                    : 'text-[#b9cacb] hover:text-[#00dbe9]'
                }`}
              >
                Quantum Mechanics
              </button>
              <button
                onClick={() => setActiveView('molecular')}
                className={`px-6 py-2 text-[12px] tracking-[0.15em] font-bold uppercase transition-all ${
                  activeView === 'molecular'
                    ? 'text-[#ddb7ff] bg-[rgba(168,85,247,0.2)] rounded'
                    : 'text-[#b9cacb] hover:text-[#ddb7ff]'
                }`}
              >
                Molecular Orbitals
              </button>
              <button
                onClick={() => setActiveView('reactivity')}
                className={`px-6 py-2 text-[12px] tracking-[0.15em] font-bold uppercase transition-all ${
                  activeView === 'reactivity'
                    ? 'text-[#ddb7ff] bg-[rgba(168,85,247,0.2)] rounded'
                    : 'text-[#b9cacb] hover:text-[#ddb7ff]'
                }`}
              >
                Reactivity
              </button>
            </div>

            <div className="grid grid-cols-3 gap-6">
              {activeView === 'quantum' && (
                <>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(0,219,233,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">SCHRÖDINGER EQUATION</div>
                    <div className="text-[20px] font-bold text-[#00dbe9] mb-2">Ĥψ = Eψ</div>
                    <div className="text-[12px] text-[#b9cacb]">Eigenvalue problem for electronic states</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(0,219,233,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">HAMILTONIAN OPERATOR</div>
                    <div className="text-[20px] font-bold text-[#00dbe9] mb-2">Ĥ = T̂ + V̂</div>
                    <div className="text-[12px] text-[#b9cacb]">Kinetic + potential energy operators</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(0,219,233,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">WAVE FUNCTION</div>
                    <div className="text-[20px] font-bold text-[#00dbe9] mb-2">ψ(r,t)</div>
                    <div className="text-[12px] text-[#b9cacb]">Complete quantum state description</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(168,85,247,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">ELECTRON DENSITY</div>
                    <div className="text-[20px] font-bold text-[#ddb7ff] mb-2">ρ(r)</div>
                    <div className="text-[12px] text-[#b9cacb]">Probability density distribution</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(168,85,247,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">EXCHANGE-CORRELATION</div>
                    <div className="text-[20px] font-bold text-[#ddb7ff] mb-2">Exc[ρ]</div>
                    <div className="text-[12px] text-[#b9cacb]">DFT energy functional</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(168,85,247,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">CORRELATION ENERGY</div>
                    <div className="text-[20px] font-bold text-[#ddb7ff] mb-2">Ecorr</div>
                    <div className="text-[12px] text-[#b9cacb]">Post-HF correction energy</div>
                  </div>
                </>
              )}

              {activeView === 'molecular' && (
                <>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(168,85,247,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">HOMO</div>
                    <div className="text-[20px] font-bold text-[#ddb7ff] mb-2">Highest Occupied</div>
                    <div className="text-[12px] text-[#b9cacb]">Donor orbital energy level</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(168,85,247,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">LUMO</div>
                    <div className="text-[20px] font-bold text-[#ddb7ff] mb-2">Lowest Unoccupied</div>
                    <div className="text-[12px] text-[#b9cacb]">Acceptor orbital energy level</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(168,85,247,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">BAND GAP</div>
                    <div className="text-[20px] font-bold text-[#ddb7ff] mb-2">Eg (eV)</div>
                    <div className="text-[12px] text-[#b9cacb]">HOMO-LUMO energy difference</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(0,219,233,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">BONDING ORBITAL</div>
                    <div className="text-[20px] font-bold text-[#00dbe9] mb-2">σ / π</div>
                    <div className="text-[12px] text-[#b9cacb]">Constructive interference</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(0,219,233,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">ANTIBONDING ORBITAL</div>
                    <div className="text-[20px] font-bold text-[#00dbe9] mb-2">σ* / π*</div>
                    <div className="text-[12px] text-[#b9cacb]">Destructive interference</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(0,219,233,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">MOLECULAR SYMMETRY</div>
                    <div className="text-[20px] font-bold text-[#00dbe9] mb-2">Point Group</div>
                    <div className="text-[12px] text-[#b9cacb]">Symmetry operations</div>
                  </div>
                </>
              )}

              {activeView === 'reactivity' && (
                <>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(0,219,233,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">ACTIVATION ENERGY</div>
                    <div className="text-[20px] font-bold text-[#00dbe9] mb-2">Ea (kcal/mol)</div>
                    <div className="text-[12px] text-[#b9cacb]">Reaction barrier height</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(0,219,233,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">REACTION ENTHALPY</div>
                    <div className="text-[20px] font-bold text-[#00dbe9] mb-2">ΔH (kcal/mol)</div>
                    <div className="text-[12px] text-[#b9cacb]">Heat of reaction</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(0,219,233,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">GIBBS FREE ENERGY</div>
                    <div className="text-[20px] font-bold text-[#00dbe9] mb-2">ΔG (kcal/mol)</div>
                    <div className="text-[12px] text-[#b9cacb]">Thermodynamic spontaneity</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(168,85,247,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">TRANSITION STATE</div>
                    <div className="text-[20px] font-bold text-[#ddb7ff] mb-2">TS</div>
                    <div className="text-[12px] text-[#b9cacb]">Maximum energy configuration</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(168,85,247,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">IRC PATH</div>
                    <div className="text-[20px] font-bold text-[#ddb7ff] mb-2">Intrinsic RC</div>
                    <div className="text-[12px] text-[#b9cacb]">Minimum energy pathway</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(168,85,247,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">FUKUI INDICES</div>
                    <div className="text-[20px] font-bold text-[#ddb7ff] mb-2">f⁺ / f⁻</div>
                    <div className="text-[12px] text-[#b9cacb]">Electrophilic/nucleophilic sites</div>
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="glass-panel p-6 rounded-xl">
            <h3 className="text-[24px] font-semibold text-[#00dbe9] mb-6">Unified API Integration</h3>
            <p className="text-[14px] text-[#b9cacb]">
              Connected to QuLab Unified API on port 8000 - POST /chemistry/synthesize
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
