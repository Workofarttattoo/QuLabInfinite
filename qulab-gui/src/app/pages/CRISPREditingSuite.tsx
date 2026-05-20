import { useNavigate } from 'react-router';
import { Image3DViewer } from '../components/Image3DViewer';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function CRISPREditingSuite() {
  const navigate = useNavigate();
  const { health, loading: healthLoading } = useLabHealth('bio');

  return (
    <>
      <style>{`
        body {
          background-color: #050505;
          color: #e5e2e1;
        }
        .active-border {
          border-color: #00dbe9;
        }
        .segmented-progress {
          display: flex;
          gap: 2px;
        }
        .segment {
          width: 8px;
          height: 12px;
          background: #1c1b1b;
        }
        .segment.active {
          background: #00dbe9;
        }
        .cursor-blink {
          border-right: 2px solid #00dbe9;
          animation: blink 1s infinite step-end;
        }
        @keyframes blink {
          0%, 100% { border-color: transparent; }
          50% { border-color: #00dbe9; }
        }
      `}</style>

      <div className="font-['JetBrains_Mono'] text-[14px] leading-[1.5] overflow-hidden min-h-screen qulab-page-bg text-foreground">
        {/* Top Navigation Shell */}
        <header className="fixed top-0 left-0 w-full z-50 flex justify-between items-center px-4 md:px-8 h-16 bg-[rgba(32,31,31,0.8)] backdrop-blur-md border-b border-[rgba(59,73,75,0.3)]">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-full bg-[rgba(0,219,233,0.2)] flex items-center justify-center border border-[rgba(0,219,233,0.4)] overflow-hidden">
              <span className="material-symbols-outlined text-[#00dbe9]">dns</span>
            </div>
            <h1 className="text-[20px] leading-[1.2] font-['Space_Grotesk'] font-bold tracking-tighter text-[#00dbe9]">GLOBAL INTELLIGENCE // ECHO v4.2</h1>
          </div>
          <div className="flex items-center gap-6">
            <div className="hidden md:flex gap-4">
              <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold text-[#dbfcff] ring-1 ring-[#00dbe9] px-2 py-1">LAB_MODE: ACTIVE</span>
              <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold text-[#b9cacb] hover:bg-white/5 transition-colors px-2 py-1 cursor-pointer">SEQUENCER_08</span>
              <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold text-[#b9cacb] hover:bg-white/5 transition-colors px-2 py-1 cursor-pointer">UNIT_8005</span>
            </div>
            <span className="material-symbols-outlined text-[#00dbe9]">sensors</span>
          </div>
        </header>

        {/* Main Workspace Canvas */}
              <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-16 pb-20 h-screen w-full grid grid-cols-12 gap-2 p-2 bg-[#050505]">
          {/* Left Sidebar: Parameters */}
          <aside className="hidden lg:flex col-span-3 flex-col gap-2">
            {/* SAAG Tile: CRISPR Protocol */}
            <div className="glass-panel p-4 flex flex-col gap-4">
              <div className="flex justify-between items-start">
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold text-[#849495]">CRISPR_PROTOCOL</span>
                <span className="w-2 h-2 rounded-full bg-[#00e639] shadow-[0_0_8px_#00e639]"></span>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between items-center border-b border-white/5 pb-2">
                  <span className="text-[#b9cacb] text-xs">GUIDE_RNA_SEQ</span>
                  <span className="text-[#00dbe9] font-['JetBrains_Mono'] text-[12px]">AUGC-GGUU-CAGG</span>
                </div>
                <div className="flex justify-between items-center border-b border-white/5 pb-2">
                  <span className="text-[#b9cacb] text-xs">CAS9_VARIANT</span>
                  <span className="text-[#00dbe9] font-['JetBrains_Mono'] text-[12px]">SP-CAS9-HF1</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#b9cacb] text-xs">TARGET_LOCUS</span>
                  <span className="text-[#00dbe9] font-['JetBrains_Mono'] text-[12px]">CHR19_P13.3</span>
                </div>
              </div>
            </div>

            {/* SAAG Tile: Editing Efficiency */}
            <div className="glass-panel p-4 flex flex-col gap-4">
              <div className="flex justify-between items-start">
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold text-[#849495]">EDITING_EFFICIENCY</span>
                <span className="bg-[#00e639] text-[#003907] text-[9px] px-1 font-bold">AUTO</span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-[24px] leading-[1] tracking-[-0.05em] font-medium font-['JetBrains_Mono'] text-[#00dbe9]">98.42</span>
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold text-[#849495]">%</span>
              </div>
              <div className="segmented-progress">
                <div className="segment active"></div>
                <div className="segment active"></div>
                <div className="segment active"></div>
                <div className="segment active"></div>
                <div className="segment active"></div>
                <div className="segment active"></div>
                <div className="segment active"></div>
                <div className="segment active"></div>
                <div className="segment active"></div>
                <div className="segment active"></div>
                <div className="segment"></div>
                <div className="segment"></div>
              </div>
            </div>

            {/* SAAG Tile: Off-Target Risk */}
            <div className="glass-panel p-4 flex flex-col gap-4 border-l-2 border-[#c10014]">
              <div className="flex justify-between items-start">
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold text-[#ffb4ab]">OFF-TARGET_RISK</span>
                <span className="w-2 h-2 rounded-full bg-[#c10014] shadow-[0_0_8px_#c10014]"></span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-[24px] leading-[1] tracking-[-0.05em] font-medium font-['JetBrains_Mono'] text-[#ffb4ab]">0.003</span>
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold text-[#849495]">EV</span>
              </div>
              <p className="text-[10px] text-[#b9cacb] leading-tight">Mismatches detected at distal sites. NIST verification required before cleavage activation.</p>
            </div>
          </aside>

          {/* Center: Helix Visualization */}
          <section className="col-span-12 lg:col-span-6 relative glass-panel overflow-hidden flex flex-col items-center justify-center group">
            {/* Header Label Overlay */}
            <div className="absolute top-4 left-4 z-10">
              <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold text-[#00dbe9] flex items-center gap-2">
                <span className="material-symbols-outlined text-[14px]">biotech</span>
                GENOMIC_VIEWPORT_PRIMARY
              </span>
            </div>

            {/* Helix Rendering Area */}
            <div className="w-full h-full relative">
              <Image3DViewer
                imageUrl="https://lh3.googleusercontent.com/aida-public/AB6AXuBvGsBjV3jnHa15wWCMSsGdV_IU0qh8-hK6nI9Mh9U-hyYivJGUa7PPPqKYuaxCbUsM2bERJIoCol8x2SYbCc2NuqXCSNDoZ2E1Xx860zsDZdc7yC4IVVIvNnbClqfw40mywlVkdKpn5PcPH1blhWQ9-jt3Z3VHf35mJaM4EbBuSb3-iDhevLntX9zLpoVOmtDiITvZpeNj4LVQ0-PdqAXHXhtOYV5ZCb27wtI4l_NDJlYEEkRoKKJuSmWJr_7L32eJkj-wsoX5Yuc"
                alt="DNA double helix with crystalline structures showing CRISPR cleavage sites"
                className="w-full h-full object-cover opacity-60"
                autoRotate={true}
              />
              {/* HUD Overlays */}
              <div className="absolute inset-0 pointer-events-none border-[1px] border-[rgba(0,219,233,0.1)] m-4 flex items-center justify-center">
                <div className="w-[80%] h-[1px] bg-gradient-to-r from-transparent via-[rgba(0,219,233,0.4)] to-transparent absolute top-1/2 -translate-y-1/2"></div>
                <div className="h-[80%] w-[1px] bg-gradient-to-b from-transparent via-[rgba(0,219,233,0.4)] to-transparent absolute left-1/2 -translate-x-1/2"></div>
              </div>
              {/* Floating Labels */}
              <div className="absolute top-1/4 right-[15%] p-2 glass-panel border-[rgba(0,219,233,0.5)] flex flex-col gap-1">
                <span className="text-[10px] font-bold text-[#00dbe9] tracking-wider">CLEAVAGE_SITE_ALPHA</span>
                <span className="text-[9px] text-[#b9cacb] font-['JetBrains_Mono']">LOC: 19:44,221,002</span>
              </div>
              <div className="absolute bottom-1/3 left-[10%] p-2 glass-panel border-[rgba(193,0,20,0.5)] flex flex-col gap-1">
                <span className="text-[10px] font-bold text-[#ffb4ab] tracking-wider">MISMATCH_WARN</span>
                <span className="text-[9px] text-[#b9cacb] font-['JetBrains_Mono']">δ: -0.042nm</span>
              </div>
            </div>

            {/* Terminal/Log Overlay */}
            <div className="absolute bottom-4 left-4 right-4 h-24 glass-panel bg-black/40 p-3 overflow-hidden border-t border-[rgba(0,219,233,0.2)]">
              <div className="flex justify-between items-center mb-1">
                <span className="text-[9px] font-bold text-[#00dbe9]">ECHO_AGI_LOG</span>
                <span className="text-[9px] text-[#849495]">PRECISION: 0.99998</span>
              </div>
              <div className="font-['JetBrains_Mono'] text-[10px] text-[#b9cacb] space-y-0.5">
                <p className="text-[#00e639]">&gt; Analyzing PAM sequence compatibility... [MATCH]</p>
                <p>&gt; Scanning for off-target binding sites... [OK]</p>
                <p>&gt; Repair template integrity verified against NIST-R2. <span className="cursor-blink"></span></p>
              </div>
            </div>
          </section>

          {/* Right Sidebar: Diagnostics & Action */}
          <aside className="hidden lg:flex col-span-3 flex-col gap-2">
            {/* SAAG Tile: NIST Verification */}
            <div className="glass-panel p-4 flex flex-col gap-4">
              <div className="flex justify-between items-start">
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold text-[#849495]">NIST_VERIFICATION</span>
                <span className="bg-[#00dbe9] text-[#00363a] text-[9px] px-1 font-bold">LOCKED</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="p-2 border border-white/5 bg-white/5 flex flex-col">
                  <span className="text-[9px] text-[#849495]">SEQUENCE</span>
                  <span className="text-[#00e639] text-xs font-bold">VALID</span>
                </div>
                <div className="p-2 border border-white/5 bg-white/5 flex flex-col">
                  <span className="text-[9px] text-[#849495]">STRUCT</span>
                  <span className="text-[#00e639] text-xs font-bold">STABLE</span>
                </div>
              </div>
            </div>

            {/* SAAG Tile: Repair Template */}
            <div className="glass-panel p-4 flex flex-col gap-4 h-full">
              <div className="flex justify-between items-start">
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold text-[#849495]">REPAIR_TEMPLATE</span>
              </div>
              <div className="flex-grow flex flex-col justify-center items-center gap-4">
                <div className="w-full aspect-video border border-dashed border-[rgba(59,73,75,0.5)] flex items-center justify-center overflow-hidden">
                  <Image3DViewer
                    imageUrl="https://lh3.googleusercontent.com/aida-public/AB6AXuDE1t5qRzbHLCpSZ7wDJpGDzq-rvdv1IWvS-hIiuEHbL7qsvXmP6jTj6QIscCn0yCTkP6Qa2B8zetMVPCT-l2NQtyS3Pvhnad6q64pmxQgf4Wt9ts8GqZgyS-ZSFJAfE-q2pBF5wdDzDbDV4_EGmjBQ2qvmqe8uQLQg-aO6zVYKmayci3PJrxHC4AjlxQqm0E5PDFHUUeq7hESRYwiPlR6W7INnB6V-SK8909WQufLXgYQLXMl-7zAtZwN28ZHGnvf5XofeFnQrxR0"
                    alt="Molecular bonds and chemical structures showing repair template"
                    className="w-full h-full object-cover opacity-40"
                    autoRotate={true}
                  />
                </div>
                <button className="w-full border border-[#00dbe9] text-[#00dbe9] py-3 text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold tracking-widest hover:bg-[rgba(0,219,233,0.1)] transition-all active:scale-95 shadow-[0_0_15px_rgba(0,219,233,0.1)]">
                  INITIATE_CLEAVAGE
                </button>
              </div>
            </div>
          </aside>
        </main>

        {/* Mobile Navigation Shell */}
        <nav className="md:hidden fixed bottom-0 left-0 w-full z-50 flex justify-around items-center px-4 h-20 bg-[rgba(28,27,27,0.9)] backdrop-blur-xl border-t border-[rgba(59,73,75,0.2)]">
          <button onClick={() => navigate('/')} className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200">
            <span className="material-symbols-outlined">grid_view</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold mt-1">TELEMETRY</span>
          </button>
          <button className="flex flex-col items-center justify-center text-[#00dbe9] border-t-2 border-[#00dbe9] pt-2 scale-95 opacity-80">
            <span className="material-symbols-outlined">hub</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold mt-1">ORCHESTRATION</span>
          </button>
          <button onClick={() => navigate('/labs')} className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200">
            <span className="material-symbols-outlined">science</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold mt-1">LABS</span>
          </button>
          <button onClick={() => navigate('/research')} className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200">
            <span className="material-symbols-outlined">monitoring</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold mt-1">RESEARCH</span>
          </button>
          <button onClick={() => navigate('/terminal')} className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200">
            <span className="material-symbols-outlined">terminal</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] font-bold mt-1">COMMAND</span>
          </button>
        </nav>

        {/* Floating Action Button */}
        <button className="fixed bottom-24 right-6 lg:bottom-10 lg:right-10 w-14 h-14 bg-[#00f0ff] text-[#006970] rounded-lg shadow-lg flex items-center justify-center border border-[#dbfcff] transition-transform active:scale-90 z-50">
          <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>bolt</span>
        </button>
      </div>
    </>
  );
}
