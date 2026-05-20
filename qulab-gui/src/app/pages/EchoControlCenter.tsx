import { useState } from 'react';
import { Link } from 'react-router';
import { useLabsHealth } from '../../lib/hooks';
import { executeEchoCommand } from '../../lib/echo-command';
interface TabType {
  id: string;
  label: string;
  icon: string;
}

const tabs: TabType[] = [
  { id: 'stream', label: 'STREAM_01', icon: 'settings_ethernet' },
  { id: 'telemetry', label: 'TELEMETRY', icon: 'analytics' },
  { id: 'comms', label: 'COMMS_LOG', icon: 'history_edu' },
  { id: 'neural', label: 'NEURAL_MAP', icon: 'psychology' },
];

interface FeedItem {
  timestamp: string;
  type: 'success' | 'info' | 'error' | 'warning';
  message: string;
}

const EchoControlCenter = () => {
  const [activeTab, setActiveTab] = useState<string>('stream');
  const [commandInput, setCommandInput] = useState<string>('');
  const [isExecuting, setIsExecuting] = useState(false);
  const { labsStatus, loading } = useLabsHealth();

  const [feedItems, setFeedItems] = useState<FeedItem[]>([
    { timestamp: '14:22:01', type: 'success', message: 'Node #NA-01 stabilized.' },
    { timestamp: '14:22:04', type: 'info', message: 'Neural cluster Alpha-9 completed cycle.' },
    { timestamp: '14:22:12', type: 'error', message: 'APAC routing delayed by 12ms.' },
    { timestamp: '14:22:18', type: 'success', message: 'Traffic shifted to EU central.' },
    { timestamp: '14:22:25', type: 'info', message: 'Batch #409-Z processing starts...' },
    { timestamp: '14:22:33', type: 'warning', message: 'Awaiting system prompt...' },
  ]);

  const getOnlineCount = () => {
    if (loading) return 0;
    return Object.values(labsStatus).filter(lab => lab.healthy).length;
  };

  const getTotalCount = () => {
    return Object.keys(labsStatus).length || 20;
  };

  const getFeedItemColor = (type: string) => {
    switch (type) {
      case 'success': return 'text-secondary-fixed-dim';
      case 'error': return 'text-error';
      case 'warning': return 'text-on-surface-variant';
      default: return 'text-surface-tint';
    }
  };

  const getFeedItemLabel = (type: string) => {
    switch (type) {
      case 'success': return 'UPLINK_SUCCESS';
      case 'error': return 'LATENCY_SPIKE';
      case 'warning': return 'AUTO_REROUTE';
      default: return 'INTEL_SYNTH';
    }
  };

  const appendFeed = (type: FeedItem['type'], message: string) => {
    const now = new Date();
    const timestamp = now.toTimeString().slice(0, 8);
    setFeedItems((prev) => [{ timestamp, type, message }, ...prev].slice(0, 24));
  };

  const handleCommandSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const text = commandInput.trim();
    if (!text || isExecuting) return;

    setIsExecuting(true);
    appendFeed('info', `CMD >> ${text}`);
    setCommandInput('');

    const outcome = await executeEchoCommand(text, {
      context: { pathname: '/echo', labSlug: 'echo', labName: 'Echo Control Center' },
    });
    appendFeed(
      outcome.ok ? 'success' : 'error',
      outcome.detail ? `${outcome.summary} — ${outcome.detail}` : outcome.summary
    );
    setIsExecuting(false);
  };

  return (
    <div className="min-h-screen bg-background text-on-surface font-body-md overflow-hidden selection:bg-surface-tint selection:text-on-primary">
      {/* Scanline Overlay */}
      <div className="fixed inset-0 pointer-events-none opacity-20 z-50"
           style={{
             background: 'linear-gradient(to bottom, transparent 50%, rgba(0, 219, 233, 0.03) 50%)',
             backgroundSize: '100% 4px'
           }}></div>

      {/* Top App Bar */}
      <header className="flex justify-between items-center w-full px-margin-mobile md:px-margin-desktop py-unit border-b border-outline-variant/50 bg-background/80 backdrop-blur-xl fixed top-0 z-40">
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-surface-tint">terminal</span>
          <h1 className="font-headline-sm text-headline-sm font-bold text-surface-tint tracking-tighter">
            QULAB_INF_OS // V.1.0.4
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-4 text-label-caps font-label-caps">
            <span className="text-primary-fixed-dim">SYSTEM_STABLE</span>
            <span className="text-on-surface-variant/40">|</span>
            <span className="text-on-surface-variant">LATENCY: 14MS</span>
          </div>
          <button className="px-3 py-1 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors">
            [ENCRYPTED]
          </button>
        </div>
      </header>

      {/* Navigation Drawer (Desktop Only) */}
      <aside className="hidden md:flex flex-col p-gutter space-y-tile-gap h-full w-80 fixed left-0 top-0 pt-20 bg-surface-container-high/80 backdrop-blur-2xl border-r border-outline-variant/20 z-30">
        <div className="px-2 mb-4">
          <h2 className="font-label-caps text-label-caps text-secondary-fixed-dim tracking-widest">
            ECHO_INTEL_OVERLAY
          </h2>
        </div>
        <nav className="flex flex-col space-y-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-4 px-4 py-3 ${
                activeTab === tab.id
                  ? 'bg-secondary-container/10 text-secondary-fixed font-bold border-l-4 border-secondary-fixed'
                  : 'text-on-surface/50 font-body-md hover:bg-on-surface/5'
              } transition-all`}
            >
              <span className="material-symbols-outlined">{tab.icon}</span>
              <span className="font-body-md text-body-md">{tab.label}</span>
            </button>
          ))}
        </nav>
        <div className="mt-auto border-t border-outline-variant/20 pt-4 px-2">
          <div className="p-3 border border-outline-variant/30"
               style={{
                 background: 'rgba(13, 18, 18, 0.7)',
                 backdropFilter: 'blur(12px)'
               }}>
            <div className="flex justify-between items-start mb-2">
              <span className="text-label-caps font-label-caps text-on-surface-variant">ACTIVE_SESSION</span>
              <span className="w-2 h-2 bg-secondary-fixed-dim rounded-full animate-pulse"></span>
            </div>
            <p className="text-[10px] text-on-surface-variant leading-tight opacity-70">
              Neural bridge established. Uplink verified via RSA-4096 cluster.
            </p>
          </div>
        </div>
      </aside>

      <main className="pt-20 pb-20 md:pb-8 md:pl-80 min-h-screen">
        <div className="p-gutter grid grid-cols-12 gap-tile-gap">

          {/* Global Operational Status / World Map */}
          <section className="col-span-12 lg:col-span-8 relative overflow-hidden group"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex justify-between items-center">
              <div className="flex items-center gap-2">
                <span className="text-label-caps font-label-caps text-surface-tint">
                  STATUS_MAP::GLOBAL_NODES
                </span>
              </div>
              <div className="flex gap-4">
                <span className="text-label-caps font-label-caps text-secondary-fixed-dim flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-secondary-fixed-dim"></span> ONLINE
                </span>
                <span className="text-label-caps font-label-caps text-on-surface-variant flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-on-surface-variant"></span> SYNCING
                </span>
              </div>
            </div>
            <div className="relative h-[340px] md:h-[450px] bg-surface-container-lowest/50">
              <img
                alt="Global world map with data points"
                className="w-full h-full object-cover opacity-40 mix-blend-screen"
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuACz78O4FoIaEtcOJDp_BhL0m1x9Zi4xxBYQS-aBzq3jCz_v-oyBWRuQ43fWzS0ZPAzZJjX3FveWK_j_MoKtyoXFfuxTqIlxh6O7sUvJI1PpUX4wBbCfjRRb4LNsldB2ABFSzr2up32bkxLoVv7sKG3UW7bNdtsSV2fu2CbkwaSkCiYrdnQG-50HPxB3Ylfue3L09TerjFjBZHoEMumBYbGdbe2tKx-sl_L_kXn7xdBvbX8Lqh4fBb319EWdi_VQXz0zRJdefEB3Ds"
              />
              {/* Node Callouts */}
              <div className="absolute top-[25%] left-[20%]">
                <div className="w-3 h-3 bg-surface-tint rounded-full"
                     style={{
                       boxShadow: '0 0 10px rgba(0, 219, 233, 0.2)',
                       borderColor: 'rgba(0, 219, 233, 0.6)'
                     }}></div>
                <div className="absolute top-4 left-4 px-2 py-1 border border-surface-tint/30"
                     style={{
                       background: 'rgba(13, 18, 18, 0.7)',
                       backdropFilter: 'blur(12px)'
                     }}>
                  <span className="text-label-caps font-label-caps text-surface-tint">NA_HUB_ALPHA</span>
                </div>
              </div>
              <div className="absolute top-[30%] left-[48%]">
                <div className="w-3 h-3 bg-surface-tint rounded-full"
                     style={{
                       boxShadow: '0 0 10px rgba(0, 219, 233, 0.2)',
                       borderColor: 'rgba(0, 219, 233, 0.6)'
                     }}></div>
                <div className="absolute top-4 left-4 px-2 py-1 border border-surface-tint/30"
                     style={{
                       background: 'rgba(13, 18, 18, 0.7)',
                       backdropFilter: 'blur(12px)'
                     }}>
                  <span className="text-label-caps font-label-caps text-surface-tint">EU_CENTRAL_ARRAY</span>
                </div>
              </div>
              <div className="absolute top-[55%] left-[80%]">
                <div className="w-3 h-3 bg-surface-tint rounded-full"
                     style={{
                       boxShadow: '0 0 10px rgba(0, 219, 233, 0.2)',
                       borderColor: 'rgba(0, 219, 233, 0.6)'
                     }}></div>
                <div className="absolute top-4 left-4 px-2 py-1 border border-surface-tint/30"
                     style={{
                       background: 'rgba(13, 18, 18, 0.7)',
                       backdropFilter: 'blur(12px)'
                     }}>
                  <span className="text-label-caps font-label-caps text-surface-tint">APAC_OFFSHORE_NODE</span>
                </div>
              </div>
              {/* Overlay Data Streams */}
              <div className="absolute bottom-4 right-4 text-right">
                <div className="font-data-display text-data-display text-surface-tint mb-1">
                  STABILITY: 99.98%
                </div>
                <div className="font-label-caps text-label-caps text-on-surface-variant">
                  ACTIVE_TRANSFERS: 4,029/SEC
                </div>
              </div>
            </div>
          </section>

          {/* SAAG Tiles Stack */}
          <div className="col-span-12 lg:col-span-4 flex flex-col gap-tile-gap">
            {/* Fleet Utilization */}
            <article className="p-4 flex flex-col justify-between bg-primary-container/5"
                     style={{
                       background: 'rgba(13, 18, 18, 0.7)',
                       backdropFilter: 'blur(12px)',
                       border: '0.5px solid rgba(0, 219, 233, 0.6)',
                       boxShadow: '0 0 10px rgba(0, 219, 233, 0.2)'
                     }}>
              <div className="flex justify-between items-start">
                <span className="font-label-caps text-label-caps text-on-surface-variant">
                  FLEET_UTILIZATION
                </span>
                <span className="material-symbols-outlined text-surface-tint">rocket_launch</span>
              </div>
              <div className="mt-4">
                <div className="font-data-display text-data-display text-surface-tint">84.2%</div>
                <div className="w-full h-2 bg-outline-variant/30 mt-2 flex gap-1">
                  {Array.from({ length: 10 }).map((_, i) => (
                    <div
                      key={i}
                      className={`h-full w-[10%] ${i < 8 ? 'bg-surface-tint' : 'bg-outline-variant/30'}`}
                    ></div>
                  ))}
                </div>
              </div>
            </article>

            {/* Qubit Stability */}
            <article className="p-4 flex flex-col justify-between border border-secondary-fixed/30 bg-secondary-container/5"
                     style={{
                       background: 'rgba(13, 18, 18, 0.7)',
                       backdropFilter: 'blur(12px)'
                     }}>
              <div className="flex justify-between items-start">
                <span className="font-label-caps text-label-caps text-on-surface-variant">
                  QUBIT_STABILITY
                </span>
                <span className="material-symbols-outlined text-secondary-fixed-dim">bolt</span>
              </div>
              <div className="mt-4">
                <div className="font-data-display text-data-display text-secondary-fixed-dim">OPTIMAL</div>
                <div className="text-label-caps font-label-caps text-secondary-fixed-dim/60 mt-1">
                  COHERENCE: 412MS
                </div>
              </div>
            </article>

            {/* Synthesis Progress */}
            <article className="p-4 flex flex-col justify-between"
                     style={{
                       background: 'rgba(13, 18, 18, 0.7)',
                       backdropFilter: 'blur(12px)',
                       border: '0.5px solid rgba(185, 202, 203, 0.15)'
                     }}>
              <div className="flex justify-between items-start">
                <span className="font-label-caps text-label-caps text-on-surface-variant">
                  SYNTHESIS_PROGRESS
                </span>
                <span className="material-symbols-outlined text-on-surface-variant">rebase_edit</span>
              </div>
              <div className="mt-4 flex items-end justify-between">
                <div>
                  <div className="font-data-display text-data-display text-on-surface">62.11%</div>
                  <div className="text-label-caps font-label-caps text-on-surface-variant mt-1">
                    BATCH_ID: #409-Z
                  </div>
                </div>
                <div className="w-12 h-12 border-2 border-dashed border-outline-variant/50 rounded-full flex items-center justify-center">
                  <span className="material-symbols-outlined animate-spin text-surface-tint">
                    progress_activity
                  </span>
                </div>
              </div>
            </article>
          </div>

          {/* Echo AGI Command Interface */}
          <section className="col-span-12 md:col-span-7 flex flex-col"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center gap-3">
              <div className="w-2 h-6 bg-surface-tint"></div>
              <h3 className="font-headline-sm text-headline-sm">ECHO_AGI::COMMAND_DIRECTIVES</h3>
            </div>
            <div className="p-gutter space-y-4">
              <div className="bg-surface-container-lowest p-4 border-l-4 border-surface-tint">
                <div className="text-label-caps font-label-caps text-surface-tint mb-2">
                  ACTIVE_MISSION: NEURAL_RECLAMATION
                </div>
                <p className="font-body-md text-on-surface/80">
                  Coordinate decentralized synthesis across all APAC offshore clusters. Prioritize data
                  integrity over latency for batch #409-Z. Grounded intelligence suggests local atmospheric
                  interference at North America Hub Alpha; rerouting telemetry through EU Central Array.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 border border-outline-variant/30">
                  <span className="text-label-caps font-label-caps text-on-surface-variant block mb-1">
                    INTEL_CONFIDENCE
                  </span>
                  <div className="text-headline-sm font-data-display">97.4%</div>
                </div>
                <div className="p-3 border border-outline-variant/30">
                  <span className="text-label-caps font-label-caps text-on-surface-variant block mb-1">
                    RISK_PARAMETER
                  </span>
                  <div className="text-headline-sm font-data-display text-error">NOMINAL</div>
                </div>
              </div>
              <form onSubmit={handleCommandSubmit} className="relative group">
                <input
                  className="w-full bg-surface-variant/10 border border-outline-variant/50 p-4 font-body-md focus:border-surface-tint focus:ring-0 placeholder:text-on-surface-variant/30 disabled:opacity-60"
                  placeholder="EXECUTE COMMAND_PROMPT..."
                  type="text"
                  value={commandInput}
                  onChange={(e) => setCommandInput(e.target.value)}
                  disabled={isExecuting}
                  autoComplete="off"
                />
                <button
                  type="submit"
                  disabled={isExecuting || !commandInput.trim()}
                  className="absolute right-4 top-1/2 -translate-y-1/2 material-symbols-outlined text-surface-tint disabled:opacity-30"
                  aria-label="Execute command"
                >
                  {isExecuting ? 'hourglass_empty' : 'keyboard_return'}
                </button>
              </form>
            </div>
          </section>

          {/* Live Intelligence Feed */}
          <section className="col-span-12 md:col-span-5 flex flex-col h-[400px]"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex justify-between items-center">
              <span className="text-label-caps font-label-caps text-on-surface-variant">
                LIVE_TACTICAL_FEED
              </span>
              <span className="text-label-caps font-label-caps text-surface-tint">REAL_TIME</span>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3 font-body-md text-[12px]">
              {feedItems.map((item, index) => (
                <div key={index} className="flex gap-3">
                  <span className="text-on-surface-variant/40">[{item.timestamp}]</span>
                  <span className={getFeedItemColor(item.type)}>{getFeedItemLabel(item.type)}</span>
                  <span className="text-on-surface-variant">{item.message}</span>
                </div>
              ))}
            </div>
            <div className="p-2 border-t border-outline-variant/30 bg-surface-container-low/50">
              <div className="flex justify-center items-center gap-4 py-2">
                <div className="w-1 h-1 bg-surface-tint rounded-full animate-ping"></div>
                <span className="text-label-caps font-label-caps text-surface-tint tracking-[0.3em]">
                  PROCESSING DATA STREAM
                </span>
              </div>
            </div>
          </section>

          {/* MCP Connections / Lab Status Grid */}
          <section className="col-span-12 mt-4">
            <div className="mb-4 flex items-center gap-2">
              <span className="material-symbols-outlined text-surface-tint">hub</span>
              <h3 className="font-label-caps text-label-caps text-surface-tint">
                MCP_INTEGRATIONS // CONNECTED_LABS
              </h3>
            </div>
            <div className="p-4 border border-outline-variant/30"
                 style={{
                   background: 'rgba(13, 18, 18, 0.7)',
                   backdropFilter: 'blur(12px)'
                 }}>
              <div className="flex justify-between items-center mb-4">
                <div className="flex gap-4">
                  <div className="flex flex-col items-end">
                    <span className="font-label-caps text-on-surface-variant">CONFIGURED_LABS</span>
                    <span className="font-data-display text-data-display text-surface-tint">
                      {getTotalCount()}
                    </span>
                  </div>
                  <div className="w-[1px] h-12 bg-outline-variant/30"></div>
                  <div className="flex flex-col items-end">
                    <span className="font-label-caps text-on-surface-variant">REACHABLE_NODES</span>
                    <span className="font-data-display text-data-display text-secondary-fixed-dim">
                      {getOnlineCount()}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-secondary-fixed-dim rounded-full"
                       style={{ boxShadow: '0 0 8px rgba(0, 230, 57, 0.6)' }}></div>
                  <span className="text-secondary-fixed-dim font-bold">STABLE // UPLINK_LOCKED</span>
                </div>
              </div>
              <div className="h-2 w-full bg-surface-variant overflow-hidden">
                <div
                  className="h-full bg-secondary-fixed-dim"
                  style={{
                    width: `${(getOnlineCount() / getTotalCount()) * 100}%`,
                    backgroundImage: 'repeating-linear-gradient(90deg, #00e639, #00e639 8px, transparent 8px, transparent 10px)'
                  }}
                ></div>
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* Bottom Navigation Bar (Mobile Only) */}
      <nav className="md:hidden fixed bottom-0 left-0 w-full z-50 flex justify-around items-stretch h-16 bg-surface-container-lowest/90 backdrop-blur-md border-t border-outline-variant/50">
        <Link
          to="/"
          className="flex flex-col items-center justify-center text-primary-fixed-dim bg-primary-container/20 border-t-2 border-primary-fixed-dim py-unit px-4"
        >
          <span className="material-symbols-outlined">grid_view</span>
          <span className="font-label-caps text-label-caps">DASHBOARD</span>
        </Link>
        <Link
          to="/labs"
          className="flex flex-col items-center justify-center text-on-surface-variant/60 py-unit px-4 hover:text-primary-fixed-dim hover:bg-surface-variant/30"
        >
          <span className="material-symbols-outlined">science</span>
          <span className="font-label-caps text-label-caps">UNITS</span>
        </Link>
        <button className="flex flex-col items-center justify-center text-on-surface-variant/60 py-unit px-4 hover:text-primary-fixed-dim hover:bg-surface-variant/30">
          <span className="material-symbols-outlined">assignment_late</span>
          <span className="font-label-caps text-label-caps">MISSION</span>
        </button>
        <button className="flex flex-col items-center justify-center text-on-surface-variant/60 py-unit px-4 hover:text-primary-fixed-dim hover:bg-surface-variant/30">
          <span className="material-symbols-outlined">settings_input_component</span>
          <span className="font-label-caps text-label-caps">SYSTEM</span>
        </button>
      </nav>
    </div>
  );
};

export { EchoControlCenter };
