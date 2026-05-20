import { useState } from 'react';
import { useNavigate } from 'react-router';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function AgentTelemetryDeepDive() {
  const navigate = useNavigate();
  const { health, loading: healthLoading } = useLabHealth('global');
  const [filterQuery, setFilterQuery] = useState('');

  const agents = [
    { id: 'AGENT_084', status: 'NOMINAL', latency: '12ms', sync: '99.8%', selected: true, error: false },
    { id: 'AGENT_112', status: 'ACTIVE', latency: '45ms', sync: '84.2%', selected: false, error: false },
    { id: 'AGENT_004', status: 'DEGRADED', latency: '892ms', sync: '12.1%', selected: false, error: true },
  ];

  const subProcesses = [
    { name: 'SYNAPTIC_MAP_V2', status: 'RUNNING', description: 'Mapping dense neural pathways for spatial reasoning nodes.', active: true },
    { name: 'LATTICE_STABILIZATION', status: 'ACTIVE', description: 'Preventing cognitive drift in long-term reasoning clusters.', active: true },
    { name: 'VECTOR_DEFRAG', status: 'PAUSED', description: 'Optimization task queued for low-latency periods.', active: false },
  ];

  const telemetryLogs = [
    { time: '08:42:11.002', level: 'INIT', message: 'AGENT_084 handshake established.', color: 'primary' },
    { time: '08:42:11.105', level: 'SYS', message: 'Validating neural lattice parameters...', color: 'normal' },
    { time: '08:42:11.230', level: 'DATA', message: 'Synaptic_map_v2: Hash verification [OK]', color: 'success' },
    { time: '08:42:11.442', level: 'PROC', message: 'Loading sub-process: Lattice_stabilization...', color: 'normal' },
    { time: '08:42:11.445', level: 'AGENT_112', message: 'Querying global vector store...', color: 'normal' },
    { time: '08:42:12.119', level: 'EXCEPTION', message: 'Retry sequence initiated at index 0x00F2', color: 'normal' },
    { time: '08:42:12.200', level: 'SYS', message: 'Recalibrating probabilistic weights...', color: 'normal' },
    { time: '08:42:12.551', level: 'AGENT_084', message: 'Pattern recognition spike detected (+14.2%)', color: 'primary' },
    { time: '08:42:12.990', level: 'PROC', message: 'Synaptic flow optimized at level 4.', color: 'normal' },
    { time: '08:42:13.102', level: 'LOG', message: 'Secure channel verified via quantum mesh.', color: 'success' },
    { time: '08:42:13.115', level: 'AGENT_084', message: 'Executing task [GRID_RECON_ALPHA]', color: 'normal' },
    { time: '08:42:13.440', level: 'TRACE', message: 'Memory allocation shift to node_099', color: 'normal' },
    { time: '08:42:13.552', level: 'SYS', message: 'Heartbeat signal [CONFIRMED]', color: 'normal' },
    { time: '08:42:14.001', level: 'AGENT_112', message: 'Latency compensation active.', color: 'primary' },
    { time: '08:42:14.223', level: 'PROC', message: 'Neural load balancing in progress...', color: 'normal' },
    { time: '08:42:14.881', level: 'WARN', message: 'Packet drop detected on cluster_C4', color: 'error' },
    { time: '08:42:15.105', level: 'SYS', message: 'Rerouting telemetry through mesh_tunnel_09', color: 'normal' },
    { time: '08:42:15.442', level: 'DATA', message: 'Lattice_stabilization: Converged at 0.002 epsilon', color: 'success' },
    { time: '08:42:15.660', level: 'AGENT_084', message: 'Telemetry burst transmission... [COMPLETE]', color: 'normal' },
  ];

  return (
    <>
      <style>{`
        ::-webkit-scrollbar {
          width: 4px;
          height: 4px;
        }
        ::-webkit-scrollbar-track {
          background: #0e0e0e;
        }
        ::-webkit-scrollbar-thumb {
          background: #3b494b;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: #00dbe9;
        }
        .scanline {
          background: linear-gradient(to bottom, rgba(0, 219, 233, 0.03) 50%, rgba(0, 0, 0, 0) 50%);
          background-size: 100% 4px;
        }
      `}</style>

      <div className="min-h-screen qulab-page-bg text-foreground font-['JetBrains_Mono'] selection:bg-[#00f0ff] selection:text-[#00363a] grid-bg">
        {/* TopAppBar */}
        <header className="fixed top-0 w-full z-50 flex justify-between items-center px-4 md:px-8 h-16 bg-[rgba(14,14,14,0.8)] backdrop-blur-xl border-b border-[rgba(59,73,75,0.3)]">
          <div className="flex items-center gap-4">
            <span className="material-symbols-outlined text-[#00dbe9]">grid_view</span>
            <h1 className="font-['Space_Grotesk'] text-[20px] leading-[1.2] tracking-tighter text-[#00dbe9]">HIVE MIND // INTEL MESH</h1>
          </div>
          <div className="flex items-center gap-4">
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#00dbe9] px-3 py-1 border border-[rgba(0,219,233,0.3)] bg-[rgba(0,240,255,0.1)]">SYSTEM LOCKED</span>
          </div>
        </header>

        {/* NavigationDrawer (Desktop) */}
        <aside className="hidden lg:flex flex-col fixed left-0 top-16 bottom-0 z-40 bg-[rgba(32,31,31,0.95)] backdrop-blur-md border-r border-[rgba(59,73,75,0.2)] w-64 p-2">
          <div className="p-4 mb-4">
            <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[rgba(185,202,203,0.5)]">TACTICAL_OS_V2</span>
          </div>
          <nav className="flex flex-col gap-1">
            <a onClick={() => navigate('/')} className="flex items-center gap-3 px-4 py-3 text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors cursor-pointer">
              <span className="material-symbols-outlined">visibility</span>
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">OVERWATCH</span>
            </a>
            <a onClick={() => navigate('/lab-wall')} className="flex items-center gap-3 px-4 py-3 text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors cursor-pointer">
              <span className="material-symbols-outlined">science</span>
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">LAB_STATUS</span>
            </a>
            <a className="flex items-center gap-3 px-4 py-3 text-[#00dbe9] bg-[rgba(0,240,255,0.1)] border-l-4 border-[#00dbe9] translate-x-1 transition-transform">
              <span className="material-symbols-outlined">leak_add</span>
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">AGENT_TX</span>
            </a>
            <a onClick={() => navigate('/hive-mind')} className="flex items-center gap-3 px-4 py-3 text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors cursor-pointer">
              <span className="material-symbols-outlined">location_searching</span>
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">GRID_MAP</span>
            </a>
            <a className="flex items-center gap-3 px-4 py-3 text-[rgba(185,202,203,0.7)] hover:bg-[rgba(0,219,233,0.05)] transition-colors cursor-pointer">
              <span className="material-symbols-outlined">lock_open</span>
              <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em]">DECRYPTION</span>
            </a>
          </nav>
          <div className="mt-auto p-4 border-t border-[rgba(59,73,75,0.2)]">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-[#353534] border border-[#00dbe9] flex items-center justify-center">
                <span className="material-symbols-outlined text-[16px] text-[#00dbe9]">account_circle</span>
              </div>
              <div className="flex flex-col">
                <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#00dbe9]">ROOT_USER</span>
                <span className="text-[9px] text-[#b9cacb]">LEVEL 7 CLEARANCE</span>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
              <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-16 pb-20 lg:pb-0 lg:ml-64 px-4 md:px-8 min-h-screen">
          <div className="max-w-7xl mx-auto py-2 grid grid-cols-12 gap-2">
            {/* Agent Filter Bar */}
            <div className="col-span-12 h-14 bg-[rgba(28,27,27,0.5)] backdrop-blur-md border border-[rgba(59,73,75,0.3)] flex items-center px-4 gap-4 mb-2">
              <span className="material-symbols-outlined text-[#00dbe9]">search</span>
              <input
                value={filterQuery}
                onChange={(e) => setFilterQuery(e.target.value)}
                className="bg-transparent border-none focus:ring-0 w-full font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#e5e2e1] placeholder:text-[rgba(185,202,203,0.3)] outline-none"
                placeholder="FILTER_BY_AGENT_ID"
                type="text"
              />
              <div className="flex items-center gap-2">
                <span className="px-2 py-1 bg-[rgba(0,240,255,0.1)] border border-[rgba(0,219,233,0.2)] text-[#00dbe9] text-[10px] font-['JetBrains_Mono'] tracking-[0.1em]">LIVE_STREAM</span>
                <div className="w-2 h-2 rounded-full bg-[#00e639] animate-pulse"></div>
              </div>
            </div>

            {/* Agent Roster (Sidebar Left) */}
            <div className="col-span-12 md:col-span-3 flex flex-col gap-2">
              <div className="bg-[rgba(28,27,27,0.5)] backdrop-blur-md border border-[rgba(59,73,75,0.3)] p-2 flex flex-col gap-2">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[rgba(185,202,203,0.6)]">ACTIVE_AGENTS</span>
                  <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#00dbe9]">08/12</span>
                </div>
                {agents.map((agent) => (
                  <div
                    key={agent.id}
                    className={`p-3 relative overflow-hidden group cursor-pointer ${
                      agent.selected
                        ? 'bg-[rgba(0,240,255,0.05)] border border-[rgba(0,219,233,0.4)]'
                        : agent.error
                        ? 'bg-[rgba(147,0,10,0.1)] border border-[rgba(255,180,171,0.4)]'
                        : 'bg-[rgba(53,53,52,0.2)] border border-[rgba(59,73,75,0.3)]'
                    } ${agent.selected ? '' : 'hover:bg-[rgba(255,255,255,0.05)]'}`}
                  >
                    {agent.selected && (
                      <div className="absolute top-0 right-0 w-8 h-8 bg-[rgba(0,219,233,0.1)] transform rotate-45 translate-x-4 -translate-y-4"></div>
                    )}
                    <div className="flex justify-between items-start mb-2">
                      <span className={`font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] ${
                        agent.selected ? 'text-[#00dbe9]' : agent.error ? 'text-[#ffb4ab]' : 'text-[#b9cacb]'
                      }`}>
                        {agent.id}
                      </span>
                      <span className={`text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] ${
                        agent.error ? 'text-[#ffb4ab] animate-pulse' : 'text-[#00e639]'
                      }`}>
                        {agent.status}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] text-[#b9cacb]">
                      <div>LATENCY: <span className={agent.error ? 'text-[#ffb4ab]' : 'text-[#e5e2e1]'}>{agent.latency}</span></div>
                      <div>SYNC: <span className={agent.error ? 'text-[#ffb4ab]' : 'text-[#e5e2e1]'}>{agent.sync}</span></div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Neural Load Tile */}
              <div className="bg-[rgba(28,27,27,0.5)] backdrop-blur-md border border-[rgba(59,73,75,0.3)] p-2">
                <div className="flex justify-between items-center mb-4">
                  <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#b9cacb]">NEURAL_LOAD</span>
                  <span className="material-symbols-outlined text-[16px] text-[#00dbe9]">analytics</span>
                </div>
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between text-[10px] font-['JetBrains_Mono'] tracking-[0.1em]">
                    <span className="text-[rgba(185,202,203,0.5)]">COGNITIVE_OVERHEAD</span>
                    <span className="text-[#00dbe9]">74%</span>
                  </div>
                  <div className="h-2 bg-[#353534] overflow-hidden flex gap-[2px]">
                    <div className="h-full bg-[#00dbe9] w-1/4"></div>
                    <div className="h-full bg-[#00dbe9] w-1/4"></div>
                    <div className="h-full bg-[#00dbe9] w-1/4"></div>
                    <div className="h-full bg-[rgba(32,31,31,0.5)] w-1/4"></div>
                  </div>
                  <div className="flex justify-between text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] mt-2">
                    <span className="text-[rgba(185,202,203,0.5)]">LATTICE_STRESS</span>
                    <span className="text-[#00e639]">21%</span>
                  </div>
                  <div className="h-2 bg-[#353534] overflow-hidden flex gap-[2px]">
                    <div className="h-full bg-[#00e639] w-[21%]"></div>
                    <div className="h-full bg-[rgba(32,31,31,0.5)] flex-1"></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Main Data Feed (Center) */}
            <div className="col-span-12 md:col-span-6 h-[calc(100vh-280px)] md:h-[calc(100vh-180px)] bg-[#0e0e0e] border border-[rgba(59,73,75,0.4)] flex flex-col relative overflow-hidden">
              <div className="scanline absolute inset-0 pointer-events-none opacity-20"></div>
              <div className="flex items-center justify-between p-3 border-b border-[rgba(59,73,75,0.2)] bg-[#1c1b1b]">
                <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#b9cacb] flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-[#00dbe9]"></span>
                  RAW_TELEMETRY_LOGS
                </span>
                <span className="font-['JetBrains_Mono'] text-[10px] text-[rgba(185,202,203,0.4)]">BUFFER: 1024KB</span>
              </div>
              <div className="flex-1 overflow-y-auto p-4 font-['JetBrains_Mono'] text-[12px] leading-relaxed">
                <div className="flex flex-col gap-1 text-[rgba(185,202,203,0.8)]">
                  {telemetryLogs.map((log, i) => (
                    <p
                      key={i}
                      className={
                        log.color === 'primary'
                          ? 'text-[rgba(0,219,233,0.6)]'
                          : log.color === 'success'
                          ? 'text-[rgba(0,230,57,0.8)]'
                          : log.color === 'error'
                          ? 'text-[rgba(255,180,171,0.7)]'
                          : ''
                      }
                    >
                      [{log.time}] <span className={log.color === 'primary' ? 'text-[#00dbe9]' : ''}>{log.level}</span> :: {log.message}
                    </p>
                  ))}
                </div>
              </div>
              <div className="h-10 bg-[#1c1b1b] border-t border-[rgba(59,73,75,0.2)] flex items-center px-4 justify-between">
                <div className="flex gap-4">
                  <span className="text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] text-[rgba(185,202,203,0.4)] hover:text-[#00dbe9] cursor-pointer transition-colors">CLEAR_BUFFER</span>
                  <span className="text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] text-[rgba(185,202,203,0.4)] hover:text-[#00dbe9] cursor-pointer transition-colors">DOWNLOAD_LOGS</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] text-[rgba(185,202,203,0.4)]">MODE:</span>
                  <span className="text-[10px] font-['JetBrains_Mono'] tracking-[0.1em] text-[#00dbe9]">RECURSIVE_ANALYTICS</span>
                </div>
              </div>
            </div>

            {/* Sub-Process Tracking (Sidebar Right) */}
            <div className="col-span-12 md:col-span-3 flex flex-col gap-2">
              {/* Sub-process List */}
              <div className="bg-[rgba(28,27,27,0.5)] backdrop-blur-md border border-[rgba(59,73,75,0.3)] p-2">
                <div className="flex justify-between items-center mb-4">
                  <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#b9cacb]">SUB_PROCESSES</span>
                  <span className="material-symbols-outlined text-[16px] text-[#00dbe9]">account_tree</span>
                </div>
                <div className="flex flex-col gap-2">
                  {subProcesses.map((proc, i) => (
                    <div key={i} className="group">
                      <div className="flex justify-between text-[11px] font-['JetBrains_Mono'] tracking-[0.1em] mb-1">
                        <span className={proc.active ? 'text-[#e5e2e1] group-hover:text-[#00dbe9] transition-colors' : 'text-[rgba(185,202,203,0.4)]'}>
                          {proc.name}
                        </span>
                        <span className={proc.active ? 'text-[#00e639]' : 'text-[rgba(185,202,203,0.4)]'}>{proc.status}</span>
                      </div>
                      <div className={`text-[9px] leading-tight ${proc.active ? 'text-[rgba(185,202,203,0.5)]' : 'text-[rgba(185,202,203,0.3)]'} ${i < subProcesses.length - 1 ? 'mb-2' : ''}`}>
                        {proc.description}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Visualization Placeholder */}
              <div className="bg-[rgba(28,27,27,0.5)] backdrop-blur-md border border-[rgba(59,73,75,0.3)] p-0 overflow-hidden h-40 relative">
                <div className="absolute inset-0 flex items-center justify-center opacity-30">
                  <span className="material-symbols-outlined text-[80px] text-[rgba(0,219,233,0.2)]">hub</span>
                </div>
                <div className="absolute inset-0 flex flex-col justify-between p-2">
                  <span className="font-['JetBrains_Mono'] text-[11px] leading-[1] tracking-[0.1em] text-[#b9cacb]">MESH_TOPOLOGY</span>
                  <div className="flex gap-1 items-end h-16">
                    <div className="w-2 bg-[rgba(0,219,233,0.4)] h-[20%]"></div>
                    <div className="w-2 bg-[rgba(0,219,233,0.6)] h-[45%]"></div>
                    <div className="w-2 bg-[#00dbe9] h-[80%]"></div>
                    <div className="w-2 bg-[rgba(0,219,233,0.5)] h-[30%]"></div>
                    <div className="w-2 bg-[rgba(0,219,233,0.8)] h-[65%]"></div>
                    <div className="w-2 bg-[#00dbe9] h-[90%]"></div>
                    <div className="w-2 bg-[rgba(0,219,233,0.4)] h-[25%]"></div>
                    <div className="w-2 bg-[rgba(0,219,233,0.6)] h-[50%]"></div>
                  </div>
                </div>
              </div>

              {/* System Visual */}
              <div className="bg-[rgba(0,240,255,0.1)] border border-[rgba(0,219,233,0.2)] p-2 flex flex-col gap-2">
                <div className="w-full aspect-video overflow-hidden">
                  <img
                    alt="Digital security grid"
                    className="w-full h-full object-cover opacity-80 mix-blend-screen"
                    src="https://lh3.googleusercontent.com/aida-public/AB6AXuAwEQ5jlaA623xFLHrSmzgwx7X_ZsYnFS8GTpukzCdNwTbAAYkY5L0eY0K0VQ7DxdhtdCZQkJaCCciNaGp4C5ng_LiQTQBTC1UvTy1VA6gKkUZ7ImlGmbttq_aNhnvZSd5x209hSnAyA2YSa8JyqChDyp9HlTW9j_IpOgw638do_M9OHHMENduZtp7D_jbirLBqosf1rWEwSnM2jtzzGM-iN5JBY7Pxqibhb7ZNqjqv0uNr_TiiAs37SmwWOotZRblQYrNNTx5Db70"
                  />
                </div>
                <span className="font-['JetBrains_Mono'] text-[10px] text-[rgba(0,219,233,0.6)]">QULAB_INFINITE // AGENT_CORE_SURVEILLANCE</span>
              </div>
            </div>
          </div>
        </main>

        {/* BottomNavBar (Mobile) */}
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
