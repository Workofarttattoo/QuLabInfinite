import { useNavigate } from 'react-router';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function GlobalIntelDashboard() {
  const navigate = useNavigate();
  const { health, loading: healthLoading } = useLabHealth('global');

  return (
    <>
      <style>{`
        body {
          min-height: max(884px, 100dvh);
          background-color: #050505;
          color: #e5e2e1;
          font-family: 'JetBrains Mono', monospace;
          overflow-x: hidden;
        }
        .scanline {
          position: absolute;
          width: 100%;
          height: 1px;
          background: linear-gradient(to right, transparent, rgba(0, 219, 233, 0.2), transparent);
          top: 0;
          z-index: 10;
        }
      `}</style>

      <div className="min-h-screen flex flex-col grid-bg">
        {/* TopAppBar */}
        <header className="bg-[rgba(32,31,31,0.8)] backdrop-blur-md text-[#00dbe9] border-b border-[rgba(59,73,75,0.3)] docked full-width top-0 flex justify-between items-center w-full px-4 md:px-8 h-16 sticky z-50">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-full border border-[#00dbe9] flex items-center justify-center overflow-hidden bg-[#353534]">
              <img
                alt="Echo Assistant"
                className="w-full h-full object-cover grayscale brightness-125"
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuA-w716MibbI_VkCFpB-wlU3UXz_VrlqbQ0EtqCOJPyDKg55k_PRneLry19aKwUdFtV5QemNfMFhFOafX82xH6If9JYpibmCx8wIzshGLvTg2Js_IeJ4hVG5OIdt4FccD3Fkf5y4VDAITrvCa0OCHfbjt8fws86AsTvnrRiBUtWdN37pAUkuyZ_AUps8P5VXl7zL_9apI0ZFE0UIUXK6V87tKGSvZEQjJYnA8gbBe4ro0flBJ-jtoS1vWnWRuX5hKOtACfWFRXnhOk"
              />
            </div>
            <h1 className="text-[20px] leading-[1.2] font-['Space_Grotesk'] font-bold tracking-tighter text-[#00dbe9] uppercase">GLOBAL INTELLIGENCE // ECHO v4.2</h1>
          </div>
          <div className="hidden md:flex gap-8">
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#dbfcff] font-bold ring-1 ring-[#00dbe9] px-3 py-1 cursor-pointer">TELEMETRY</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#b9cacb] hover:bg-white/5 transition-colors px-3 py-1 cursor-pointer">ORCHESTRATION</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#b9cacb] hover:bg-white/5 transition-colors px-3 py-1 cursor-pointer">LABS</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#b9cacb] hover:bg-white/5 transition-colors px-3 py-1 cursor-pointer">RESEARCH</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="material-symbols-outlined text-[#00dbe9]">sensors</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] hidden sm:block">STATUS: SYNCED</span>
          </div>
        </header>

        {/* Main Content Canvas */}
              <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="flex-1 p-4 md:p-8 grid grid-cols-12 gap-2 max-w-[1920px] mx-auto w-full">
          {/* Left Column: Fleet Status & Echo Intelligence */}
          <section className="col-span-12 lg:col-span-8 flex flex-col gap-2">
            {/* Fleet Status Map Card (Large SAAG Tile) */}
            <div className="glass-panel active-glow relative min-h-[400px] flex flex-col p-4 rounded-lg overflow-hidden">
              <div className="flex justify-between items-center mb-4">
                <div className="flex flex-col">
                  <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#849495]">LOCATION_MATRIX</span>
                  <h2 className="text-[20px] leading-[1.2] font-['Space_Grotesk'] text-[#e5e2e1] uppercase">GLOBAL FLEET DISPATCH</h2>
                </div>
                <div className="flex items-center gap-2">
                  <span className="bg-[rgba(0,230,57,0.2)] text-[#00e639] px-2 py-0.5 rounded text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] border border-[rgba(0,230,57,0.4)]">220 NODES ONLINE</span>
                  <span className="material-symbols-outlined text-[#00dbe9]">public</span>
                </div>
              </div>
              {/* Map Visualization Placeholder */}
              <div className="flex-1 relative bg-[rgba(14,14,14,0.5)] rounded border border-[rgba(59,73,75,0.2)] overflow-hidden">
                <img
                  alt="Fleet Map"
                  className="w-full h-full object-cover opacity-60 mix-blend-screen"
                  src="https://lh3.googleusercontent.com/aida-public/AB6AXuCV6Or_kXoLjc-RTASHGrQSTov0EuDO32Z7AQkCS-NWwoITCdYdP1hqryanWxbk_C7qMjjdZgB-yt_z3lxNHpoaSZeBsGbl9wHKPGjUAwhLOiIRtlJJTLzOsbn0aJZxkYh0ZNuatfzSFtHxXj7PFx-5c0A9xvhq0EyknF-fY5oW8o_aK2kCZ51e9FT8VrvEcn2t7H6rPXINgRZqbz4Dc8nEdq0MtbuC5wZ6AdqRLmDKzq7fPlUq1PdRq8pVcHXmv2z6galySH8OFjg"
                />
                {/* Tactical Overlays */}
                <div className="absolute inset-0 p-4 pointer-events-none">
                  <div className="grid grid-cols-4 h-full">
                    <div className="border-r border-[rgba(59,73,75,0.1)]"></div>
                    <div className="border-r border-[rgba(59,73,75,0.1)]"></div>
                    <div className="border-r border-[rgba(59,73,75,0.1)]"></div>
                  </div>
                </div>
                {/* Active Node Tooltip (Faked) */}
                <div className="absolute top-1/4 left-1/3 glass-panel p-3 rounded-sm border-[rgba(0,219,233,0.5)] shadow-2xl">
                  <div className="text-[10px] text-[#00dbe9] font-bold mb-1">NODE_SVALBARD_04</div>
                  <div className="text-[14px] font-['JetBrains_Mono']">74.23°N, 15.42°E</div>
                  <div className="mt-2 h-1 w-full bg-[#353534]">
                    <div className="h-full bg-[#00dbe9] w-[88%]"></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom Row: Labs Summary & Research Throughput */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {/* Research Throughput */}
              <div className="glass-panel p-4 rounded-lg">
                <div className="flex justify-between items-start mb-4">
                  <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#849495]">TELEMETRY_LOG</span>
                  <span className="text-[#00e639] font-bold text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] uppercase">AUTO</span>
                </div>
                <h3 className="text-[20px] leading-[1.2] font-['Space_Grotesk'] text-[#e5e2e1] mb-2">RESEARCH THROUGHPUT</h3>
                <div className="flex items-baseline gap-2 mb-4">
                  <span className="text-[48px] leading-[1.1] tracking-[-0.02em] font-['Space_Grotesk'] font-bold text-[#00dbe9]">1.42</span>
                  <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#849495]">PETAFLOPS / SEC</span>
                </div>
                <div className="flex gap-1 h-8">
                  <div className="flex-1 bg-[rgba(0,219,233,0.8)]"></div>
                  <div className="flex-1 bg-[rgba(0,219,233,0.6)]"></div>
                  <div className="flex-1 bg-[rgba(0,219,233,0.9)]"></div>
                  <div className="flex-1 bg-[rgba(0,219,233,0.4)]"></div>
                  <div className="flex-1 bg-[rgba(0,219,233,0.7)]"></div>
                  <div className="flex-1 bg-[rgba(0,219,233,0.5)]"></div>
                  <div className="flex-1 bg-[rgba(0,219,233,0.85)]"></div>
                  <div className="flex-1 bg-[rgba(0,219,233,0.3)]"></div>
                  <div className="flex-1 bg-[rgba(0,219,233,0.6)]"></div>
                  <div className="flex-1 bg-[#00dbe9] animate-pulse"></div>
                </div>
              </div>

              {/* Laboratory Status */}
              <div className="glass-panel p-4 rounded-lg">
                <div className="flex justify-between items-start mb-4">
                  <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#849495]">BIO_METRICS</span>
                  <span className="material-symbols-outlined text-[#c10014]">warning</span>
                </div>
                <h3 className="text-[20px] leading-[1.2] font-['Space_Grotesk'] text-[#e5e2e1] mb-2">CRITICAL ANOMALIES</h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between border-b border-[rgba(59,73,75,0.1)] pb-2">
                    <span className="text-[14px] leading-[1.5] font-['JetBrains_Mono']">LAB_SHANGHAI_09</span>
                    <span className="text-[#ffb4ab] font-bold text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono']">TEMP_EXCEEDED</span>
                  </div>
                  <div className="flex items-center justify-between border-b border-[rgba(59,73,75,0.1)] pb-2">
                    <span className="text-[14px] leading-[1.5] font-['JetBrains_Mono']">LAB_BOSTON_22</span>
                    <span className="text-[#00e639] font-bold text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono']">STABLE</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-[14px] leading-[1.5] font-['JetBrains_Mono']">LAB_BERLIN_01</span>
                    <span className="text-[#00dbe9] font-bold text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono']">LOCKED</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Right Column: Echo Core & Mission Intel */}
          <section className="col-span-12 lg:col-span-4 flex flex-col gap-2">
            {/* Echo Core Module */}
            <div className="glass-panel p-4 rounded-lg relative overflow-hidden group">
              <div className="scanline"></div>
              <div className="flex items-center gap-4 mb-6">
                <div className="w-16 h-16 border border-[#00dbe9] p-1">
                  <div className="w-full h-full relative overflow-hidden">
                    <img
                      alt="Echo Visualizer"
                      className="w-full h-full object-cover"
                      src="https://lh3.googleusercontent.com/aida-public/AB6AXuB5YtNLA3LpdXWrx8zYDpxyaVdnLV4pa--NzGeAxcIeEAaSSMXHXjaAWyzVAGKRjJJiLSTLTlF354Y3SxQxo0VtbddfwSFyj6gLYqFTvL5wrm5M5JmFftUKXYoME4KZnE3c8ltaFgOHe7mCJXdYobMfmj66ARTa2JsI-uW5fbuenE_9C-iXXIqmhnVDay4dT-kwig3tjawCxbHd_fs3E1ALtt2RRYbV5fLp_eedvqKysJiW7Bw-xpaL_e1cR9m817IPCulTB8xcz7w"
                    />
                  </div>
                </div>
                <div>
                  <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#00dbe9]">AI_CORE_ACTIVE</span>
                  <h2 className="text-[20px] leading-[1.2] font-['Space_Grotesk'] text-[#e5e2e1]">ECHO ANALYTICA</h2>
                </div>
              </div>
              <div className="bg-[rgba(28,27,27,0.5)] p-4 border-l-2 border-[#00dbe9] mb-4">
                <p className="text-[14px] leading-[1.5] font-['JetBrains_Mono'] italic text-[#b9cacb] leading-relaxed">
                  "Detecting suboptimal pattern emergence in Lab Sector 7. Global throughput remains within 98.4% of projected mission parameters. Awaiting command for deep-dive diagnostics."
                </p>
              </div>
              <button className="w-full py-3 bg-[rgba(0,219,233,0.1)] border border-[#00dbe9] text-[#00dbe9] font-['JetBrains_Mono'] font-bold uppercase tracking-widest hover:bg-[rgba(0,219,233,0.2)] transition-all flex items-center justify-center gap-2">
                <span className="material-symbols-outlined text-[18px]">terminal</span>
                EXECUTE_COMMAND
              </button>
            </div>

            {/* Mission Intelligence Feed */}
            <div className="glass-panel flex-1 p-4 rounded-lg flex flex-col">
              <div className="flex justify-between items-center mb-4">
                <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono'] text-[#849495]">INTEL_STREAM</span>
                <span className="text-[10px] font-['JetBrains_Mono'] animate-pulse text-[#00dbe9]">LIVE</span>
              </div>
              <div className="space-y-6 overflow-y-auto pr-2">
                {/* Intel Item 1 */}
                <div className="space-y-1">
                  <div className="flex items-center justify-between text-[10px] text-[#849495] font-['JetBrains_Mono']">
                    <span>TIMESTAMP: 14:22:01 UTC</span>
                    <span>SOURCE: NODE_21</span>
                  </div>
                  <p className="text-[14px] leading-[1.5] font-['JetBrains_Mono'] text-[#e5e2e1]">Pathogen simulation complete. Result: 92% virulence match.</p>
                  <div className="flex gap-2">
                    <span className="px-2 py-0.5 bg-[#353534] text-[10px] font-bold">BIO_HAZARD</span>
                    <span className="px-2 py-0.5 bg-[#353534] text-[10px] font-bold">L3_RESTRICTED</span>
                  </div>
                </div>
                {/* Intel Item 2 */}
                <div className="space-y-1 opacity-70">
                  <div className="flex items-center justify-between text-[10px] text-[#849495] font-['JetBrains_Mono']">
                    <span>TIMESTAMP: 14:19:54 UTC</span>
                    <span>SOURCE: ECHO_CORE</span>
                  </div>
                  <p className="text-[14px] leading-[1.5] font-['JetBrains_Mono'] text-[#e5e2e1]">Global infrastructure load-balancing initiated for node resilience.</p>
                </div>
                {/* Intel Item 3 */}
                <div className="space-y-1 opacity-50">
                  <div className="flex items-center justify-between text-[10px] text-[#849495] font-['JetBrains_Mono']">
                    <span>TIMESTAMP: 14:15:33 UTC</span>
                    <span>SOURCE: NODE_88</span>
                  </div>
                  <p className="text-[14px] leading-[1.5] font-['JetBrains_Mono'] text-[#e5e2e1]">Resource extraction protocols verified in deep-sea sector.</p>
                </div>
                {/* Intel Item 4 */}
                <div className="space-y-1 opacity-30">
                  <div className="flex items-center justify-between text-[10px] text-[#849495] font-['JetBrains_Mono']">
                    <span>TIMESTAMP: 14:10:02 UTC</span>
                    <span>SOURCE: SECURITY_MESH</span>
                  </div>
                  <p className="text-[14px] leading-[1.5] font-['JetBrains_Mono'] text-[#e5e2e1]">Perimeter breach detected in virtual environment #12. Resolved.</p>
                </div>
              </div>
              <div className="mt-auto pt-4 border-t border-[rgba(59,73,75,0.1)]">
                <div className="flex items-center gap-2 text-[#849495]">
                  <span className="w-2 h-2 rounded-full bg-[#00e639]"></span>
                  <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono']">ALL SYSTEMS NOMINAL</span>
                </div>
              </div>
            </div>
          </section>
        </main>

        {/* BottomNavBar */}
        <nav className="bg-[rgba(28,27,27,0.9)] backdrop-blur-xl text-[#00dbe9] border-t border-[rgba(59,73,75,0.2)] fixed bottom-0 left-0 w-full z-50 flex justify-around items-center px-4 h-20">
          <div className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200 cursor-pointer">
            <span className="material-symbols-outlined">grid_view</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono']">TELEMETRY</span>
          </div>
          <div className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200 cursor-pointer">
            <span className="material-symbols-outlined">hub</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono']">ORCHESTRATION</span>
          </div>
          <div className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200 cursor-pointer">
            <span className="material-symbols-outlined">science</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono']">LABS</span>
          </div>
          <div className="flex flex-col items-center justify-center text-[#849495] pt-2 hover:text-[#dbfcff] transition-all duration-200 cursor-pointer">
            <span className="material-symbols-outlined">monitoring</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono']">RESEARCH</span>
          </div>
          <div className="flex flex-col items-center justify-center text-[#00dbe9] border-t-2 border-[#00dbe9] pt-2 scale-95 opacity-80 cursor-pointer">
            <span className="material-symbols-outlined">terminal</span>
            <span className="text-[11px] leading-[1] tracking-[0.1em] font-['JetBrains_Mono']">COMMAND</span>
          </div>
        </nav>

        {/* Extra UI Deco for Tactical Authenticity */}
        <div className="fixed top-20 left-4 hidden xl:block pointer-events-none opacity-20">
          <div className="text-[8px] font-['JetBrains_Mono'] space-y-1">
            <div>S_KEY: 0x8823FB99</div>
            <div>MESH_ADDR: 192.110.0.4</div>
            <div>VOLTAGE: 12.4V</div>
            <div>SIGNAL: -44dBm</div>
          </div>
        </div>
        <div className="fixed bottom-24 right-8 z-40">
          <button className="w-14 h-14 bg-[#00dbe9] text-[#002022] flex items-center justify-center rounded-full shadow-lg hover:scale-105 active:scale-95 transition-transform">
            <span className="material-symbols-outlined" style={{fontVariationSettings: "'FILL' 1"}}>add</span>
          </button>
        </div>
      </div>
    </>
  );
}
