import { EchoLabCommandInline } from '../components/EchoLabCommandInline';
import { Navigation } from '../components/Navigation';
import { useLabHealth } from '../../lib/hooks';
import { MATERIALS_DATABASE_COUNT, MATERIALS_DATABASE_DESCRIPTION } from '../../lib/materials-constants';
import { useEffect, useState } from 'react';
import { Image3DViewer } from '../components/Image3DViewer';

export function MaterialsLab() {
  const { health, loading: healthLoading } = useLabHealth('materials');
  const [activeTab, setActiveTab] = useState<'properties' | 'search' | 'compare'>('properties');
  const [rotation, setRotation] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setRotation((r) => (r + 1) % 360);
    }, 50);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 px-[32px]">
        <div className="max-w-[1440px] mx-auto">
          <EchoLabCommandInline className="mb-8" />
          <div className="mb-12">
            <div className="mb-4 inline-flex items-center gap-2 px-3 py-1 bg-[rgba(168,85,247,0.2)] border border-[rgba(168,85,247,0.4)] rounded-full">
              <span className="w-2 h-2 rounded-full bg-[#ddb7ff] animate-pulse"></span>
              <span className="text-[12px] leading-[16px] tracking-[0.15em] font-bold text-[#ddb7ff]">MATERIALS_SCIENCE_ENGINE</span>
            </div>
            <h1 className="text-[48px] leading-[56px] tracking-[-0.02em] font-bold text-[#ddb7ff] mb-4">
              Materials Science Lab
            </h1>
            <p className="text-[18px] leading-[28px] text-[#b9cacb] max-w-3xl">
              {MATERIALS_DATABASE_DESCRIPTION} — Unified API Port 8000
            </p>
          </div>

          <div className="grid grid-cols-3 gap-6 mb-8">
            <div className="glass-panel p-6 rounded-xl neon-glow-purple">
              <h3 className="text-[24px] font-semibold text-[#ddb7ff] mb-4">Lab Status</h3>
              {healthLoading ? (
                <div className="text-[#b9cacb]">Loading...</div>
              ) : health ? (
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-3 h-3 rounded-full bg-[#ddb7ff] animate-pulse"></span>
                    <span className="text-[#ddb7ff] font-semibold">✅ {health.status || 'ONLINE'}</span>
                  </div>
                  <div className="text-[12px] text-[#b9cacb] mt-4">Unified API Connected</div>
                </div>
              ) : (
                <div className="text-[#ffb4ab]">❌ Offline</div>
              )}
            </div>

            <div className="glass-panel p-6 rounded-xl border-white/10">
              <h3 className="text-[20px] font-semibold text-[#00dbe9] mb-4">Database</h3>
              <div className="text-[32px] font-bold text-[#00dbe9]">{MATERIALS_DATABASE_COUNT}</div>
              <div className="text-[12px] text-[#b9cacb] mt-2">Materials in database</div>
            </div>

            <div className="glass-panel p-6 rounded-xl border-white/10">
              <h3 className="text-[20px] font-semibold text-[#ddb7ff] mb-4">API Endpoint</h3>
              <div className="text-[12px] text-[#00dbe9] font-mono">/materials/analyze</div>
              <div className="text-[12px] text-[#b9cacb] mt-2">POST to Unified API</div>
            </div>
          </div>

          {/* 3D Molecule Visualization */}
          <div className="grid grid-cols-2 gap-6 mb-8">
            <div className="glass-panel p-6 rounded-xl border-[#ddb7ff]/20 relative overflow-hidden">
              <div className="absolute top-4 left-4 z-10">
                <h3 className="text-[20px] font-semibold text-[#ddb7ff] mb-1">Material Structure</h3>
                <p className="text-[12px] text-[#b9cacb]">Graphene Lattice - 3D Model</p>
              </div>

              {/* 3D Rotating Molecule Visualization */}
              <div className="relative w-full aspect-square flex items-center justify-center">
                <Image3DViewer
                  imageUrl="https://lh3.googleusercontent.com/aida-public/AB6AXuAPx7KDng1WMn4mTe00YezYh3HJrzAuba-D0IhED3DlrPo0Y0x81eQsgadPfDT4WJJiMphwAJv80jgXyqc4rL3KP4-irPUT_IUPIrIGVM64_p1snb53LNZBhwZlbvHLaPz7PKpdv1SDm62gd96NJDVNS62cqybM49FUPn8Fjopx64ZDYIILxuNp7Xc37heph1n3X0KZcLcCADa7mGVEFjrjPYTdrtbcPDt9Fb5Y0NTgeKugSxOvsAJD5TJxx2b_CKQ8Qs5LpzYLoF4"
                  alt="Graphene Molecular Structure 3D"
                  className="w-full h-full opacity-90 mix-blend-screen"
                  autoRotate={true}
                />
                <div className="absolute inset-0 border border-[#ddb7ff]/20 rounded-lg pointer-events-none"></div>
              </div>

              <div className="mt-4 flex gap-2">
                <div className="flex-1 text-center p-2 bg-[#ddb7ff]/10 rounded">
                  <div className="text-[10px] text-[#b9cacb]">ATOMS</div>
                  <div className="text-[16px] font-bold text-[#ddb7ff]">240</div>
                </div>
                <div className="flex-1 text-center p-2 bg-[#00dbe9]/10 rounded">
                  <div className="text-[10px] text-[#b9cacb]">BONDS</div>
                  <div className="text-[16px] font-bold text-[#00dbe9]">360</div>
                </div>
              </div>
            </div>

            {/* Echo AI Educational Guide */}
            <div className="glass-panel p-6 rounded-xl border-[#00dbe9]/20 relative overflow-hidden">
              <div className="absolute -right-8 -top-8 w-32 h-32 bg-[#00dbe9]/10 rounded-full blur-2xl"></div>

              <div className="flex items-start gap-3 mb-4">
                <div className="w-10 h-10 rounded-full bg-[#00dbe9]/20 flex items-center justify-center">
                  <span className="material-symbols-outlined text-[#00dbe9] animate-pulse">psychology</span>
                </div>
                <div>
                  <h4 className="text-[14px] font-bold text-[#00dbe9] uppercase tracking-wider">Echo AI Guide</h4>
                  <p className="text-[10px] text-[#b9cacb]">Material Design Assistant</p>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <h5 className="text-[12px] font-semibold text-white mb-2">Understanding Graphene Properties:</h5>
                  <p className="text-[12px] text-[#b9cacb] leading-relaxed">
                    Graphene's hexagonal lattice structure gives it exceptional <span className="text-[#00dbe9] font-medium">tensile strength (130 GPa)</span> - stronger than steel.
                    The sp² bonding creates highly conductive pathways.
                  </p>
                </div>

                {/* Visual Aid Example */}
                <div className="bg-[#0a0e14]/80 p-3 rounded-lg border border-white/5">
                  <div className="text-[10px] font-bold text-[#ddb7ff] uppercase mb-2">Visual Check:</div>
                  <div className="flex gap-2 mb-2">
                    <div className="flex-1 h-12 bg-gradient-to-r from-yellow-600 to-yellow-400 rounded flex items-center justify-center text-[10px] font-bold">
                      ✓ Golden Color
                    </div>
                    <div className="flex-1 h-12 bg-gradient-to-r from-black via-orange-600 to-black rounded flex items-center justify-center text-[10px] font-bold">
                      ✗ Burnt
                    </div>
                  </div>
                  <p className="text-[10px] text-[#b9cacb]">
                    During CVD growth, a <span className="text-yellow-400">golden sheen</span> indicates proper carbon deposition.
                    Black with orange edges means <span className="text-orange-400">overheating</span> - reduce temperature.
                  </p>
                </div>

                <div className="flex gap-2">
                  <button className="flex-1 px-3 py-2 bg-[#00dbe9]/20 hover:bg-[#00dbe9]/30 rounded text-[11px] font-semibold text-[#00dbe9] transition-all">
                    Next Step →
                  </button>
                  <button className="px-3 py-2 bg-white/5 hover:bg-white/10 rounded text-[11px] text-[#b9cacb] transition-all">
                    ⓘ Info
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-panel p-8 rounded-xl">
            <div className="flex gap-4 mb-6 border-b border-white/10 pb-4">
              <button
                onClick={() => setActiveTab('properties')}
                className={`px-6 py-2 text-[12px] tracking-[0.15em] font-bold uppercase transition-all ${
                  activeTab === 'properties'
                    ? 'text-[#ddb7ff] bg-[rgba(168,85,247,0.2)] rounded'
                    : 'text-[#b9cacb] hover:text-[#ddb7ff]'
                }`}
              >
                Property Analysis
              </button>
              <button
                onClick={() => setActiveTab('search')}
                className={`px-6 py-2 text-[12px] tracking-[0.15em] font-bold uppercase transition-all ${
                  activeTab === 'search'
                    ? 'text-[#00dbe9] bg-[rgba(0,219,233,0.2)] rounded'
                    : 'text-[#b9cacb] hover:text-[#00dbe9]'
                }`}
              >
                Material Search
              </button>
              <button
                onClick={() => setActiveTab('compare')}
                className={`px-6 py-2 text-[12px] tracking-[0.15em] font-bold uppercase transition-all ${
                  activeTab === 'compare'
                    ? 'text-[#00dbe9] bg-[rgba(0,219,233,0.2)] rounded'
                    : 'text-[#b9cacb] hover:text-[#00dbe9]'
                }`}
              >
                Compare Materials
              </button>
            </div>

            <div className="grid grid-cols-2 gap-6">
              {activeTab === 'properties' && (
                <>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(168,85,247,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">TENSILE STRENGTH</div>
                    <div className="text-[20px] font-bold text-[#ddb7ff] mb-2">σ (MPa)</div>
                    <div className="text-[12px] text-[#b9cacb]">Maximum stress before failure</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(168,85,247,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">THERMAL CONDUCTIVITY</div>
                    <div className="text-[20px] font-bold text-[#ddb7ff] mb-2">κ (W/mK)</div>
                    <div className="text-[12px] text-[#b9cacb]">Heat transfer rate</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(0,219,233,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">ELECTRICAL CONDUCTIVITY</div>
                    <div className="text-[20px] font-bold text-[#00dbe9] mb-2">σₑ (S/m)</div>
                    <div className="text-[12px] text-[#b9cacb]">Charge carrier mobility</div>
                  </div>
                  <div className="p-6 bg-[rgba(13,21,21,0.4)] rounded-lg border border-[rgba(0,219,233,0.3)]">
                    <div className="text-[10px] tracking-[0.15em] text-[#b9cacb] mb-3">BAND GAP</div>
                    <div className="text-[20px] font-bold text-[#00dbe9] mb-2">Eg (eV)</div>
                    <div className="text-[12px] text-[#b9cacb]">Electronic structure</div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
