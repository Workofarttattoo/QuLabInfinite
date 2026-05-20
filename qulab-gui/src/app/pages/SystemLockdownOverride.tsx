import { useState } from 'react';
import { useNavigate } from 'react-router';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function SystemLockdownOverride() {
  const navigate = useNavigate();
  const { health, loading: healthLoading } = useLabHealth('global');
  const [overrideKey, setOverrideKey] = useState('');

  const protocols = [
    { name: 'CORE_LOGIC_FREEZE', active: false },
    { name: 'AGENT_MUTE_ALL', active: true },
    { name: 'DATABASE_AIRGAP', active: false },
  ];

  const echoReasoningLogs = [
    'Detected anomalous pattern in lattice growth, recommending immediate freeze.',
    'Probability of recursive agent feedback loop: 87.4%.',
    'Simulation stability degrading. Airgap required to prevent cross-contamination.',
  ];

  const metrics = [
    { label: 'LATTICE_VOLTAGE', value: '1.21 GW', color: 'primary' },
    { label: 'ACTIVE_AGENTS', value: '0.00 / STATIC', color: 'error' },
    { label: 'UPLINK_STABILITY', value: 'OFFLINE', color: 'success' },
    { label: 'RECOVERY_EST', value: 'T-UNKNOWN', color: 'normal' },
  ];

  return (
    <>
      <style>{`
        .scanline {
          background: linear-gradient(to bottom, transparent 50%, rgba(0, 219, 233, 0.05) 51%);
          background-size: 100% 4px;
        }
        .alert-glow {
          box-shadow: 0 0 20px rgba(193, 0, 20, 0.2);
        }
      `}</style>

      <div className="min-h-screen qulab-page-bg text-foreground font-['JetBrains_Mono'] selection:bg-[#00f0ff] selection:text-[#00363a]">
        {/* Top Navigation */}
        <header className="fixed top-0 w-full z-50 flex justify-between items-center px-4 md:px-8 h-16 bg-[rgba(14,14,14,0.8)] backdrop-blur-xl border-b border-[rgba(59,73,75,0.3)]">
          <div className="flex items-center gap-4">
            <span className="material-symbols-outlined text-[#00dbe9]">grid_view</span>
            <h1 className="font-['Space_Grotesk'] text-[20px] leading-[1.2] tracking-tighter text-[#00dbe9]">HIVE MIND // INTEL MESH</h1>
          </div>
          <div className="flex items-center gap-2">
            <span className="bg-[#c10014] text-[#ffcec8] px-3 py-1 font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] rounded-sm">SYSTEM LOCKED</span>
          </div>
        </header>

        {/* Sidebar Navigation (Desktop) */}
        <nav className="hidden lg:flex flex-col fixed left-0 top-16 bottom-0 z-40 w-64 bg-[rgba(32,31,31,0.95)] backdrop-blur-md border-r border-[rgba(59,73,75,0.2)] py-8 px-4">
          <div className="mb-8 px-4">
            <p className="font-['JetBrains_Mono'] text-[10px] text-[rgba(185,202,203,0.5)] mb-2">SESSION_ID</p>
            <p className="font-['Space_Grotesk'] text-[20px] leading-[1.2] text-[#00dbe9]">QLB-INF-009</p>
          </div>
          <ul className="space-y-1">
            <li>
              <a onClick={() => navigate('/')} className="flex items-center gap-3 px-4 py-3 font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors group cursor-pointer">
                <span className="material-symbols-outlined text-xl">visibility</span>
                OVERWATCH
              </a>
            </li>
            <li>
              <a onClick={() => navigate('/lab-wall')} className="flex items-center gap-3 px-4 py-3 font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors cursor-pointer">
                <span className="material-symbols-outlined text-xl">science</span>
                LAB_STATUS
              </a>
            </li>
            <li>
              <a className="flex items-center gap-3 px-4 py-3 font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#00dbe9] bg-[rgba(0,240,255,0.1)] border-l-4 border-[#00dbe9] translate-x-1 transition-transform">
                <span className="material-symbols-outlined text-xl">leak_add</span>
                AGENT_TX
              </a>
            </li>
            <li>
              <a onClick={() => navigate('/hive-mind')} className="flex items-center gap-3 px-4 py-3 font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors cursor-pointer">
                <span className="material-symbols-outlined text-xl">location_searching</span>
                GRID_MAP
              </a>
            </li>
            <li>
              <a className="flex items-center gap-3 px-4 py-3 font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors cursor-pointer">
                <span className="material-symbols-outlined text-xl">lock_open</span>
                DECRYPTION
              </a>
            </li>
          </ul>
        </nav>

        {/* Main Content Canvas */}
              <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="lg:pl-64 pt-24 pb-32 px-4 md:px-8 min-h-screen scanline">
          <div className="max-w-6xl mx-auto space-y-4">
            {/* Alert Header */}
            <section className="grid grid-cols-1 md:grid-cols-3 gap-2">
              <div className="md:col-span-2 glass-panel p-6 border-l-4 border-[#93000c] alert-glow">
                <div className="flex justify-between items-start mb-4">
                  <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#ffdad6] bg-[#93000a] px-2 py-0.5">CRITICAL_THREAT_DETECTED</span>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-[#ffb4ab] animate-pulse"></div>
                    <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#ffb4ab]">LEVEL_05_ALERT</span>
                  </div>
                </div>
                <h2 className="font-['Space_Grotesk'] text-[32px] leading-[1.2] mb-2 text-[#e5e2e1]">SYSTEM LOCKDOWN OVERRIDE</h2>
                <p className="text-[#b9cacb] max-w-xl">Protocol Q-INF-99 initiated. All non-essential lattice operations are suspended pending administrative authentication. Anomalous growth detected in quadrant 7-B.</p>
              </div>
              <div className="glass-panel p-6 flex flex-col justify-between border-t-4 border-[#00e639]">
                <div className="flex justify-between items-center">
                  <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#00e639]">SANDBOX_STATUS</span>
                  <span className="material-symbols-outlined text-[#00e639]">verified_user</span>
                </div>
                <div>
                  <p className="font-['JetBrains_Mono'] text-[24px] leading-[1] tracking-[-0.05em] text-[#00e639]">SIMULATED</p>
                  <p className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[rgba(185,202,203,0.5)] mt-1">NO IMPACT ON LIVE CORE</p>
                </div>
              </div>
            </section>

            {/* Main Control Grid */}
            <section className="grid grid-cols-1 md:grid-cols-12 gap-2">
              {/* Central Override Panel */}
              <div className="md:col-span-8 glass-panel p-8 flex flex-col items-center justify-center text-center space-y-8 relative overflow-hidden">
                <div className="absolute inset-0 opacity-10 pointer-events-none">
                  <div className="w-full h-full border-[20px] border-[#93000c] rotate-45 scale-150"></div>
                </div>
                <div className="w-20 h-20 rounded-full border-2 border-[#00dbe9] flex items-center justify-center mb-4">
                  <span className="material-symbols-outlined text-4xl text-[#00dbe9]">key</span>
                </div>
                <div className="w-full max-w-md space-y-4">
                  <label className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#b9cacb] block text-left">MASTER_OVERRIDE_KEY</label>
                  <div className="relative group">
                    <input
                      value={overrideKey}
                      onChange={(e) => setOverrideKey(e.target.value)}
                      className="w-full bg-[rgba(255,255,255,0.05)] border border-[rgba(59,73,75,0.3)] p-4 font-['JetBrains_Mono'] text-[#00dbe9] placeholder:text-[rgba(185,202,203,0.2)] focus:outline-none focus:ring-1 focus:ring-[#00dbe9] transition-all"
                      placeholder="•••• •••• •••• ••••"
                      type="password"
                    />
                    <div className="absolute right-4 top-1/2 -translate-y-1/2 flex gap-2">
                      <span className="w-1 h-4 bg-[rgba(0,219,233,0.3)]"></span>
                      <span className="w-1 h-4 bg-[rgba(0,219,233,0.3)]"></span>
                    </div>
                  </div>
                  <button className="w-full bg-[#93000a] hover:bg-[#93000c] text-[#ffdad6] font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] py-6 border border-[rgba(255,180,171,0.2)] transition-all hover:scale-[0.99] active:scale-95 group relative">
                    <span className="relative z-10">INITIATE_SYSTEM_LOCKDOWN</span>
                    <div className="absolute inset-0 bg-[rgba(255,255,255,0.05)] opacity-0 group-hover:opacity-100 transition-opacity"></div>
                  </button>
                </div>
                <p className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#ffdad6] tracking-widest">AUTHORIZED ACCESS ONLY // BIOMETRIC_BYPASS_DISABLED</p>
              </div>

              {/* Reasoning Log & Protocols */}
              <div className="md:col-span-4 space-y-2">
                {/* Echo Reasoning Log */}
                <div className="glass-panel p-4 h-full border-l border-[rgba(0,219,233,0.3)]">
                  <div className="flex items-center gap-2 mb-4 border-b border-[rgba(59,73,75,0.2)] pb-2">
                    <span className="material-symbols-outlined text-[#00dbe9] text-sm">psychology</span>
                    <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#00dbe9]">ECHO_REASONING_LOG</span>
                  </div>
                  <div className="space-y-4">
                    {echoReasoningLogs.map((log, i) => (
                      <div key={i} className="flex gap-3">
                        <span className="text-[#00dbe9] font-bold mt-1">&gt;&gt;</span>
                        <p className="text-sm text-[#b9cacb]">{log}</p>
                      </div>
                    ))}
                  </div>
                  <div className="mt-8 pt-4 border-t border-[rgba(59,73,75,0.2)]">
                    <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[rgba(185,202,203,0.4)]">PROTOCOL_STACK</span>
                    <div className="mt-4 space-y-2">
                      {protocols.map((protocol, i) => (
                        <div
                          key={i}
                          className={`flex justify-between items-center glass-panel p-3 border-l-2 ${
                            protocol.active ? 'border-[#849495]' : 'border-[#93000c]'
                          }`}
                        >
                          <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">{protocol.name}</span>
                          <span className={`material-symbols-outlined text-sm ${protocol.active ? 'text-[#00e639]' : 'text-[#ffb4ab]'}`}>
                            {protocol.active ? 'check' : 'close'}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* Bottom Data Feed */}
            <section className="grid grid-cols-1 md:grid-cols-4 gap-2">
              {metrics.map((metric, i) => (
                <div key={i} className="glass-panel p-4">
                  <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[rgba(185,202,203,0.5)]">{metric.label}</span>
                  <p
                    className={`font-['JetBrains_Mono'] text-[24px] leading-[1] tracking-[-0.05em] ${
                      metric.color === 'primary'
                        ? 'text-[#00dbe9]'
                        : metric.color === 'error'
                        ? 'text-[#ffdad6]'
                        : metric.color === 'success'
                        ? 'text-[#00e639]'
                        : 'text-[#e5e2e1]'
                    }`}
                  >
                    {metric.value}
                  </p>
                </div>
              ))}
            </section>
          </div>
        </main>

        {/* Bottom Navigation Bar (Mobile) */}
        <nav className="md:hidden fixed bottom-0 w-full z-50 flex justify-around items-stretch h-16 bg-[rgba(14,14,14,0.9)] backdrop-blur-2xl border-t border-[rgba(59,73,75,0.3)]">
          <a className="flex flex-col items-center justify-center text-[rgba(185,202,203,0.6)] px-4 py-2 hover:text-[#00dbe9] hover:bg-[rgba(255,255,255,0.05)] cursor-pointer">
            <span className="material-symbols-outlined">map</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">FLEET</span>
          </a>
          <a className="flex flex-col items-center justify-center text-[rgba(185,202,203,0.6)] px-4 py-2 hover:text-[#00dbe9] hover:bg-[rgba(255,255,255,0.05)] cursor-pointer">
            <span className="material-symbols-outlined">lan</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">MESH</span>
          </a>
          <a className="flex flex-col items-center justify-center text-[#00dbe9] bg-[rgba(0,240,255,0.2)] border-t-2 border-[#00dbe9] px-4 py-2 scale-95 transition-transform duration-100">
            <span className="material-symbols-outlined">terminal</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">LOGS</span>
          </a>
          <a className="flex flex-col items-center justify-center text-[rgba(185,202,203,0.6)] px-4 py-2 hover:text-[#00dbe9] hover:bg-[rgba(255,255,255,0.05)] cursor-pointer">
            <span className="material-symbols-outlined">sync_alt</span>
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">SYNC</span>
          </a>
        </nav>
      </div>
    </>
  );
}
