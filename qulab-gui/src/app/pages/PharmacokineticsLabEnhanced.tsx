import { useNavigate } from 'react-router';
import { Image3DViewer } from '../components/Image3DViewer';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function PharmacokineticsLabEnhanced() {
  const navigate = useNavigate();
  const { health, loading, error } = useLabHealth('drug');

  return (
    <div className="bg-[#131313] text-[#e5e2e1] font-['JetBrains_Mono'] selection:bg-[#00f0ff]/30 min-h-screen overflow-hidden flex flex-col">
      <style>
        {`
          .active-border-cyan {
            border-color: #00f0ff;
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
          }
          .terminal-cursor::after {
            content: '|';
            animation: blink 1s step-end infinite;
          }
          @keyframes blink { 50% { opacity: 0; } }
        `}
      </style>

      {/* TopAppBar */}
      <header className="sticky top-0 z-50 bg-[#131313]/80 backdrop-blur-xl border-b border-[#3b494b]/30 flex justify-between items-center w-full px-4 md:px-8 h-16">
        <div className="flex items-center gap-4">
          <span className="material-symbols-outlined text-[#dbfcff]">security</span>
          <h1 className="text-[20px] font-bold text-[#dbfcff] tracking-tighter font-['Space_Grotesk']">QULAB_INFINITE_OS</h1>
        </div>
        <div className="hidden md:flex gap-8 items-center h-full">
          <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495] hover:text-[#00f0ff] transition-colors cursor-pointer" onClick={() => navigate('/')}>DASHBOARD</span>
          <span className="text-[11px] font-bold tracking-[0.1em] text-[#dbfcff] border-b-2 border-[#dbfcff] h-full flex items-center">LABS</span>
          <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495] hover:text-[#00f0ff] transition-colors cursor-pointer">MISSION</span>
          <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495] hover:text-[#00f0ff] transition-colors cursor-pointer">SYSTEM</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="material-symbols-outlined text-[#dbfcff]">sensors</span>
        </div>
      </header>

      {/* Main Content Layout */}
            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="flex-grow grid grid-cols-12 gap-2 p-4 md:p-8 overflow-hidden max-h-[calc(100vh-128px)] grid-bg">
        {/* Left Panel: Technical Metrics & Reasoning */}
        <aside className="col-span-12 lg:col-span-3 flex flex-col gap-2">
          {/* Binding Affinity Tile */}
          <section className="glass-panel p-4 flex flex-col gap-2 relative overflow-hidden">
            <div className="flex justify-between items-start">
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495]">METRIC_01</span>
              <span className="w-2 h-2 rounded-full bg-[#00f0ff]"></span>
            </div>
            <h3 className="text-[11px] font-bold tracking-[0.1em] text-[#dbfcff]">BINDING AFFINITY</h3>
            <div className="flex items-baseline gap-2">
              <span className="text-[24px] leading-none tracking-[-0.05em] font-medium text-[#dbfcff]">10</span>
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495]">nM [K_D]</span>
            </div>
            <div className="mt-4 flex gap-1 h-2">
              <div className="flex-1 bg-[#00f0ff]"></div>
              <div className="flex-1 bg-[#00f0ff]"></div>
              <div className="flex-1 bg-[#00f0ff]"></div>
              <div className="flex-1 bg-[#00f0ff]/20"></div>
            </div>
          </section>

          {/* Echo Reasoning Log */}
          <section className="glass-panel p-4 flex-grow flex flex-col gap-2 border-l-2 border-[#dbfcff]">
            <div className="flex justify-between items-start">
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495]">ECHO_SYSTEM_LOG</span>
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#72ff70]">ACTIVE</span>
            </div>
            <div className="overflow-y-auto pr-2 space-y-4">
              <p className="text-[14px] leading-relaxed text-[#b9cacb]">
                <span className="text-[#dbfcff] font-bold">REASONING:</span>
                Molecular docking simulations indicate a high-affinity bond at the receptor cleft. The observed K_D of 10nM suggests significant thermodynamic stability.
              </p>
              <p className="text-[14px] leading-relaxed text-[#b9cacb]">
                <span className="text-[#dbfcff] font-bold">CLEARANCE:</span>
                Hepatic clearance rates are projected within the 0.5 - 0.8 L/hr range. Initial half-life prediction (T1/2) is stable at 4.2 hours under current titration parameters.
              </p>
              <div className="p-2 bg-[#201f1f] text-[#dbfcff] text-[10px] font-bold tracking-[0.1em] terminal-cursor">
                EXECUTING SEQUENCE_BETA_V3...
              </div>
            </div>
          </section>
        </aside>

        {/* Center Panel: 3D Visualizer */}
        <section className="col-span-12 lg:col-span-6 flex flex-col gap-2">
          <div className="glass-panel relative flex-grow overflow-hidden border-[#dbfcff]/40 active-border-cyan rounded-none">
            <div className="absolute top-4 left-4 z-10">
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#dbfcff] bg-[#dbfcff]/10 px-2 py-1">REAL_TIME_VIZ</span>
            </div>
            <div className="absolute top-4 right-4 z-10 flex gap-2">
              <div className="px-2 py-1 bg-[#72ff70]/20 border border-[#72ff70] text-[#72ff70] text-[10px] font-bold tracking-[0.1em]">AUTO</div>
              <div className="px-2 py-1 bg-[#3b494b]/30 border border-[#849495] text-[#849495] text-[10px] font-bold tracking-[0.1em]">LOCKED</div>
            </div>

            {/* Main 3D Visualizer Content */}
            <Image3DViewer
              imageUrl="https://lh3.googleusercontent.com/aida-public/AB6AXuCumBxbU992HNpwoBxqncprK3JwH19TfXohARzDJKywZkpPb3Wv_sZgDFRN2c9U_RIOLj9OKr0S2Uc1VVckTL5SJSagWi6iX0RdWFzGhiWWgLbECKCtptndbkz5IKSGrzlzAbAiomlRXkaccTr8JzGieAnMCz7jGlEwXl7MTDloWzl19-IjilfYPXZ6xh1zvou7kaAtTjkZc3bO8WvqsIFgBU3xz4Z8l_RZKSC0Cf-EsnTlUg1XMo2fCF0ytmWGP1yFEnJsMwcKNuQ"
              alt="Drug-receptor interaction visualization"
              className="w-full h-full"
              autoRotate={true}
            />

            {/* SVG Overlays for tactical feel */}
            <div className="absolute inset-0 pointer-events-none border-[0.5px] border-[#dbfcff]/20 m-4 flex flex-col justify-between">
              <div className="flex justify-between p-2">
                <div className="w-4 h-4 border-t border-l border-[#dbfcff]"></div>
                <div className="w-4 h-4 border-t border-r border-[#dbfcff]"></div>
              </div>
              <div className="flex justify-center mb-8">
                <div className="flex flex-col items-center">
                  <div className="w-[1px] h-32 bg-gradient-to-t from-[#dbfcff]/50 to-transparent"></div>
                  <span className="text-[9px] font-bold tracking-[0.1em] text-[#dbfcff] mt-2">AXIS_Y_LOCKED</span>
                </div>
              </div>
              <div className="flex justify-between p-2">
                <div className="w-4 h-4 border-b border-l border-[#dbfcff]"></div>
                <div className="w-4 h-4 border-b border-r border-[#dbfcff]"></div>
              </div>
            </div>

            <div className="absolute bottom-4 left-4 right-4 glass-panel p-2 flex justify-between items-center border-none bg-[#131313]/40">
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495]">FRAME: 89002 // FR: 60FPS</span>
              <div className="flex items-center gap-4">
                <span className="material-symbols-outlined text-[#dbfcff] scale-75">videocam</span>
                <span className="material-symbols-outlined text-[#dbfcff] scale-75">settings_overscan</span>
                <span className="material-symbols-outlined text-[#ffb4ab] scale-75">fiber_manual_record</span>
              </div>
            </div>
          </div>
        </section>

        {/* Right Panel: Data Telemetry */}
        <aside className="col-span-12 lg:col-span-3 flex flex-col gap-2">
          {/* Concentration Tile */}
          <section className="glass-panel p-4 flex flex-col gap-2">
            <div className="flex justify-between items-start">
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495]">METRIC_02</span>
              <span className="w-2 h-2 rounded-full bg-[#72ff70]"></span>
            </div>
            <h3 className="text-[11px] font-bold tracking-[0.1em] text-[#dbfcff]">CONCENTRATION</h3>
            <div className="flex items-baseline gap-2">
              <span className="text-[24px] leading-none tracking-[-0.05em] font-medium text-[#dbfcff]">200</span>
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495]">mg/L</span>
            </div>
            {/* Mini Sparkline-style chart */}
            <div className="mt-2 h-16 w-full flex items-end gap-1">
              <div className="bg-[#dbfcff]/40 w-full" style={{ height: '40%' }}></div>
              <div className="bg-[#dbfcff]/40 w-full" style={{ height: '55%' }}></div>
              <div className="bg-[#dbfcff]/60 w-full" style={{ height: '70%' }}></div>
              <div className="bg-[#00f0ff] w-full" style={{ height: '85%' }}></div>
              <div className="bg-[#dbfcff]/50 w-full" style={{ height: '40%' }}></div>
              <div className="bg-[#dbfcff]/30 w-full" style={{ height: '20%' }}></div>
            </div>
          </section>

          {/* Half-Life Tile */}
          <section className="glass-panel p-4 flex flex-col gap-2">
            <div className="flex justify-between items-start">
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495]">METRIC_03</span>
              <span className="material-symbols-outlined text-[#dbfcff] text-xs">shutter_speed</span>
            </div>
            <h3 className="text-[11px] font-bold tracking-[0.1em] text-[#dbfcff]">PREDICTED HALF-LIFE</h3>
            <div className="flex items-baseline gap-2">
              <span className="text-[24px] leading-none tracking-[-0.05em] font-medium text-[#dbfcff]">4.2</span>
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495]">HOURS [T1/2]</span>
            </div>
            <div className="mt-2 border-t border-[#3b494b]/30 pt-2">
              <div className="flex justify-between text-[10px] font-bold tracking-[0.1em] text-[#849495]">
                <span>LOWER_BOUND</span>
                <span>3.8H</span>
              </div>
              <div className="flex justify-between text-[10px] font-bold tracking-[0.1em] text-[#849495]">
                <span>UPPER_BOUND</span>
                <span>4.5H</span>
              </div>
            </div>
          </section>

          {/* Clearance Rate Tile */}
          <section className="glass-panel p-4 flex flex-col gap-2">
            <div className="flex justify-between items-start">
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495]">METRIC_04</span>
              <span className="material-symbols-outlined text-[#dbfcff] text-xs">water_drop</span>
            </div>
            <h3 className="text-[11px] font-bold tracking-[0.1em] text-[#dbfcff]">CLEARANCE RATE</h3>
            <div className="flex items-baseline gap-2">
              <span className="text-[24px] leading-none tracking-[-0.05em] font-medium text-[#dbfcff]">0.68</span>
              <span className="text-[11px] font-bold tracking-[0.1em] text-[#849495]">L/hr</span>
            </div>
          </section>

          {/* Status Cluster */}
          <section className="flex-grow glass-panel p-4 flex flex-col justify-end">
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-none border border-[#dbfcff] flex items-center justify-center">
                  <span className="material-symbols-outlined text-[#dbfcff] text-sm">microbiology</span>
                </div>
                <div>
                  <p className="text-[10px] font-bold tracking-[0.1em] text-[#849495]">BIOSYNC</p>
                  <p className="text-[11px] font-bold tracking-[0.1em] text-[#dbfcff]">99.8% READY</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-none border border-[#849495] flex items-center justify-center">
                  <span className="material-symbols-outlined text-[#849495] text-sm">database</span>
                </div>
                <div>
                  <p className="text-[10px] font-bold tracking-[0.1em] text-[#849495]">DATA_FEED</p>
                  <p className="text-[11px] font-bold tracking-[0.1em] text-[#849495]">ENCRYPTED</p>
                </div>
              </div>
              <button className="w-full mt-4 py-3 border border-[#dbfcff] text-[#dbfcff] text-[11px] font-bold tracking-[0.1em] hover:bg-[#dbfcff]/10 active:scale-95 transition-all">
                INITIATE_PHASE_04
              </button>
            </div>
          </section>
        </aside>
      </main>

      {/* BottomNavBar */}
      <nav className="fixed bottom-0 w-full z-50 bg-[#131313]/90 backdrop-blur-2xl border-t border-[#3b494b]/50 flex justify-around items-center h-16 px-2">
        <a className="flex flex-col items-center justify-center text-[#849495] h-full px-4 hover:bg-[#2a2a2a]/50 transition-colors cursor-pointer" onClick={() => navigate('/')}>
          <span className="material-symbols-outlined">grid_view</span>
          <span className="text-[11px] font-bold tracking-[0.1em]">DASHBOARD</span>
        </a>
        <a className="flex flex-col items-center justify-center text-[#dbfcff] bg-[#dbfcff]/10 border-t-2 border-[#dbfcff] h-full px-4 scale-95 duration-75">
          <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>biotech</span>
          <span className="text-[11px] font-bold tracking-[0.1em]">LABS</span>
        </a>
        <a className="flex flex-col items-center justify-center text-[#849495] h-full px-4 hover:bg-[#2a2a2a]/50 transition-colors cursor-pointer">
          <span className="material-symbols-outlined">target</span>
          <span className="text-[11px] font-bold tracking-[0.1em]">MISSION</span>
        </a>
        <a className="flex flex-col items-center justify-center text-[#849495] h-full px-4 hover:bg-[#2a2a2a]/50 transition-colors cursor-pointer">
          <span className="material-symbols-outlined">terminal</span>
          <span className="text-[11px] font-bold tracking-[0.1em]">SYSTEM</span>
        </a>
      </nav>
    </div>
  );
}
