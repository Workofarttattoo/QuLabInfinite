import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { apiClient } from '../../lib/api-client';
import { useLabsHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';
import { AppBottomNav } from '../components/AppBottomNav';

interface GlobalStatus {
  stability: number;
  activeTransfers: number;
  nodes: Array<{
    id: string;
    label: string;
    top: string;
    left: string;
  }>;
}

interface MetricData {
  fleetUtilization: number;
  qubitStability: string;
  coherence: number;
  synthesisProgress: number;
  batchId: string;
}

interface TacticalFeed {
  timestamp: string;
  type: string;
  message: string;
}

export function GlobalDashboardOS() {
  const navigate = useNavigate();
  const { labsStatus } = useLabsHealth();
  const [globalStatus, setGlobalStatus] = useState<GlobalStatus>({
    stability: 99.98,
    activeTransfers: 4029,
    nodes: [
      { id: 'NA_HUB_ALPHA', label: 'NA_HUB_ALPHA', top: '25%', left: '20%' },
      { id: 'EU_CENTRAL_ARRAY', label: 'EU_CENTRAL_ARRAY', top: '30%', left: '48%' },
      { id: 'APAC_OFFSHORE_NODE', label: 'APAC_OFFSHORE_NODE', top: '55%', left: '80%' }
    ]
  });

  const [metrics, setMetrics] = useState<MetricData>({
    fleetUtilization: 84.2,
    qubitStability: 'OPTIMAL',
    coherence: 412,
    synthesisProgress: 62.11,
    batchId: '#409-Z'
  });

  const [tacticalFeed, setTacticalFeed] = useState<TacticalFeed[]>([
    { timestamp: '14:22:01', type: 'UPLINK_SUCCESS', message: 'Node #NA-01 stabilized.' },
    { timestamp: '14:22:04', type: 'INTEL_SYNTH', message: 'Neural cluster Alpha-9 completed cycle.' },
    { timestamp: '14:22:12', type: 'LATENCY_SPIKE', message: 'APAC routing delayed by 12ms.' },
    { timestamp: '14:22:18', type: 'AUTO_REROUTE', message: 'Traffic shifted to EU central.' },
    { timestamp: '14:22:25', type: 'INTEL_SYNTH', message: 'Batch #409-Z processing starts...' },
    { timestamp: '14:22:33', type: 'SYSTEM', message: 'Awaiting system prompt...' }
  ]);

  const [systemLatency, setSystemLatency] = useState(14);

  useEffect(() => {
    // Fetch global system status
    const fetchGlobalStatus = async () => {
      try {
        const status = await apiClient.getGlobalSystemStatus();
        // Update metrics based on backend response
        const onlineCount = Object.values(labsStatus).filter(s => s.healthy).length;
        setSystemLatency(Math.random() * 10 + 10); // Simulated latency
      } catch (error) {
        console.error('Failed to fetch global status:', error);
      }
    };

    fetchGlobalStatus();
    const interval = setInterval(fetchGlobalStatus, 15000);
    return () => clearInterval(interval);
  }, [labsStatus]);

  useEffect(() => {
    // Simulate live feed updates
    const interval = setInterval(() => {
      const now = new Date();
      const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
      const types = ['INTEL_SYNTH', 'UPLINK_SUCCESS', 'AUTO_REROUTE', 'SYSTEM'];
      const messages = [
        'System checkpoint validated.',
        'Neural processing cycle complete.',
        'Batch synchronization in progress.',
        'Network optimization detected.'
      ];

      setTacticalFeed(prev => [
        ...prev.slice(-5),
        {
          timestamp: timeStr,
          type: types[Math.floor(Math.random() * types.length)],
          message: messages[Math.floor(Math.random() * messages.length)]
        }
      ]);
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  const getFeedTypeColor = (type: string) => {
    switch (type) {
      case 'UPLINK_SUCCESS':
      case 'AUTO_REROUTE':
        return 'text-secondary-fixed-dim';
      case 'INTEL_SYNTH':
        return 'text-surface-tint';
      case 'LATENCY_SPIKE':
        return 'text-error';
      default:
        return 'text-on-surface-variant';
    }
  };

  return (
    <div className="min-h-screen bg-background text-on-surface font-body-md overflow-hidden selection:bg-surface-tint selection:text-on-primary">
      {/* Scanline overlay */}
      <div className="fixed inset-0 scanline pointer-events-none opacity-20 z-50"></div>

      {/* Top App Bar */}
      <header className="flex justify-between items-center w-full px-margin-mobile md:px-margin-desktop py-unit border-b border-outline-variant/50 bg-background/80 backdrop-blur-xl fixed top-0 z-40">
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-surface-tint">terminal</span>
          <h1 className="font-headline-sm text-headline-sm font-bold text-surface-tint tracking-tighter">QULAB_INF_OS // V.1.0.4</h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-4 text-label-caps font-label-caps">
            <span className="text-primary-fixed-dim">SYSTEM_STABLE</span>
            <span className="text-on-surface-variant/40">|</span>
            <span className="text-on-surface-variant">LATENCY: {systemLatency.toFixed(0)}MS</span>
          </div>
          <button className="px-3 py-1 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors">
            [ENCRYPTED]
          </button>
        </div>
      </header>

      {/* Navigation Drawer (Desktop Only) */}
      <aside className="hidden md:flex flex-col p-gutter space-y-tile-gap h-full w-80 fixed left-0 top-0 pt-20 bg-surface-container-high/80 backdrop-blur-2xl border-r border-outline-variant/20 z-30">
        <div className="px-2 mb-4">
          <h2 className="font-label-caps text-label-caps text-secondary-fixed-dim tracking-widest">ECHO_INTEL_OVERLAY</h2>
        </div>
        <nav className="flex flex-col space-y-1">
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-4 px-4 py-3 bg-secondary-container/10 text-secondary-fixed font-bold border-l-4 border-secondary-fixed hover:bg-on-surface/5 transition-all"
          >
            <span className="material-symbols-outlined">settings_ethernet</span>
            <span className="font-body-md text-body-md">STREAM_01</span>
          </button>
          <button
            onClick={() => navigate('/telemetry')}
            className="flex items-center gap-4 px-4 py-3 text-on-surface/50 font-body-md hover:bg-on-surface/5 transition-all"
          >
            <span className="material-symbols-outlined">analytics</span>
            <span className="font-body-md text-body-md">TELEMETRY</span>
          </button>
          <button
            onClick={() => navigate('/comms')}
            className="flex items-center gap-4 px-4 py-3 text-on-surface/50 font-body-md hover:bg-on-surface/5 transition-all"
          >
            <span className="material-symbols-outlined">history_edu</span>
            <span className="font-body-md text-body-md">COMMS_LOG</span>
          </button>
          <button
            onClick={() => navigate('/neural')}
            className="flex items-center gap-4 px-4 py-3 text-on-surface/50 font-body-md hover:bg-on-surface/5 transition-all"
          >
            <span className="material-symbols-outlined">psychology</span>
            <span className="font-body-md text-body-md">NEURAL_MAP</span>
          </button>
        </nav>
        <div className="mt-auto border-t border-outline-variant/20 pt-4 px-2">
          <div className="glass-panel p-3 border border-outline-variant/30">
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

      {/* Main Content Canvas */}
            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-20 pb-20 md:pb-8 md:pl-80 min-h-screen">
        <div className="p-gutter grid grid-cols-12 gap-tile-gap">
          {/* Global Operational Status / World Map */}
          <section className="col-span-12 lg:col-span-8 glass-panel relative overflow-hidden group">
            <div className="p-4 border-b border-outline-variant/30 flex justify-between items-center">
              <div className="flex items-center gap-2">
                <span className="text-label-caps font-label-caps text-surface-tint">STATUS_MAP::GLOBAL_NODES</span>
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
              {globalStatus.nodes.map(node => (
                <div key={node.id} className="absolute group/node" style={{ top: node.top, left: node.left }}>
                  <div className="w-3 h-3 bg-surface-tint rounded-full border-glow-cyan"></div>
                  <div className="absolute top-4 left-4 glass-panel px-2 py-1 border border-surface-tint/30">
                    <span className="text-label-caps font-label-caps text-surface-tint">{node.label}</span>
                  </div>
                </div>
              ))}
              {/* Overlay Data Streams */}
              <div className="absolute bottom-4 right-4 text-right">
                <div className="font-data-display text-data-display text-surface-tint mb-1">
                  STABILITY: {globalStatus.stability}%
                </div>
                <div className="font-label-caps text-label-caps text-on-surface-variant">
                  ACTIVE_TRANSFERS: {globalStatus.activeTransfers.toLocaleString()}/SEC
                </div>
              </div>
            </div>
          </section>

          {/* SAAG Tiles Stack */}
          <div className="col-span-12 lg:col-span-4 flex flex-col gap-tile-gap">
            {/* Fleet Utilization */}
            <article className="glass-panel p-4 flex flex-col justify-between border-glow-cyan bg-primary-container/5">
              <div className="flex justify-between items-start">
                <span className="font-label-caps text-label-caps text-on-surface-variant">FLEET_UTILIZATION</span>
                <span className="material-symbols-outlined text-surface-tint">rocket_launch</span>
              </div>
              <div className="mt-4">
                <div className="font-data-display text-data-display text-surface-tint">{metrics.fleetUtilization}%</div>
                <div className="w-full h-2 bg-outline-variant/30 mt-2 flex gap-1">
                  {[...Array(10)].map((_, i) => (
                    <div
                      key={i}
                      className={`h-full ${i < Math.floor(metrics.fleetUtilization / 10) ? 'bg-surface-tint' : 'bg-outline-variant/30'} w-[10%]`}
                    ></div>
                  ))}
                </div>
              </div>
            </article>

            {/* Qubit Stability */}
            <article className="glass-panel p-4 flex flex-col justify-between border border-secondary-fixed/30 bg-secondary-container/5">
              <div className="flex justify-between items-start">
                <span className="font-label-caps text-label-caps text-on-surface-variant">QUBIT_STABILITY</span>
                <span className="material-symbols-outlined text-secondary-fixed-dim">bolt</span>
              </div>
              <div className="mt-4">
                <div className="font-data-display text-data-display text-secondary-fixed-dim">{metrics.qubitStability}</div>
                <div className="text-label-caps font-label-caps text-secondary-fixed-dim/60 mt-1">
                  COHERENCE: {metrics.coherence}MS
                </div>
              </div>
            </article>

            {/* Synthesis Progress */}
            <article className="glass-panel p-4 flex flex-col justify-between">
              <div className="flex justify-between items-start">
                <span className="font-label-caps text-label-caps text-on-surface-variant">SYNTHESIS_PROGRESS</span>
                <span className="material-symbols-outlined text-on-surface-variant">rebase_edit</span>
              </div>
              <div className="mt-4 flex items-end justify-between">
                <div>
                  <div className="font-data-display text-data-display text-on-surface">{metrics.synthesisProgress}%</div>
                  <div className="text-label-caps font-label-caps text-on-surface-variant mt-1">BATCH_ID: {metrics.batchId}</div>
                </div>
                <div className="w-12 h-12 border-2 border-dashed border-outline-variant/50 rounded-full flex items-center justify-center">
                  <span className="material-symbols-outlined animate-spin text-surface-tint">progress_activity</span>
                </div>
              </div>
            </article>
          </div>

          {/* Echo AGI Command Interface */}
          <section className="col-span-12 md:col-span-7 glass-panel flex flex-col">
            <div className="p-4 border-b border-outline-variant/30 flex items-center gap-3">
              <div className="w-2 h-6 bg-surface-tint"></div>
              <h3 className="font-headline-sm text-headline-sm">ECHO_AGI::COMMAND_DIRECTIVES</h3>
            </div>
            <div className="p-gutter space-y-4">
              <div className="bg-surface-container-lowest p-4 border-l-4 border-surface-tint">
                <div className="text-label-caps font-label-caps text-surface-tint mb-2">ACTIVE_MISSION: NEURAL_RECLAMATION</div>
                <p className="font-body-md text-on-surface/80">
                  Coordinate decentralized synthesis across all APAC offshore clusters. Prioritize data integrity over latency for batch {metrics.batchId}. Grounded intelligence suggests local atmospheric interference at North America Hub Alpha; rerouting telemetry through EU Central Array.
                </p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="p-3 border border-outline-variant/30">
                  <span className="text-label-caps font-label-caps text-on-surface-variant block mb-1">INTEL_CONFIDENCE</span>
                  <div className="text-headline-sm font-data-display">97.4%</div>
                </div>
                <div className="p-3 border border-outline-variant/30">
                  <span className="text-label-caps font-label-caps text-on-surface-variant block mb-1">RISK_PARAMETER</span>
                  <div className="text-headline-sm font-data-display text-error">NOMINAL</div>
                </div>
              </div>
              <div className="relative group">
                <input
                  className="w-full bg-surface-variant/10 border border-outline-variant/50 p-4 font-body-md focus:border-surface-tint focus:ring-0 placeholder:text-on-surface-variant/30"
                  placeholder="EXECUTE COMMAND_PROMPT..."
                  type="text"
                />
                <span className="absolute right-4 top-1/2 -translate-y-1/2 material-symbols-outlined text-surface-tint">keyboard_return</span>
              </div>
            </div>
          </section>

          {/* Live Intelligence Feed */}
          <section className="col-span-12 md:col-span-5 glass-panel flex flex-col h-[400px]">
            <div className="p-4 border-b border-outline-variant/30 flex justify-between items-center">
              <span className="text-label-caps font-label-caps text-on-surface-variant">LIVE_TACTICAL_FEED</span>
              <span className="text-label-caps font-label-caps text-surface-tint">REAL_TIME</span>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3 font-body-md text-[12px]">
              {tacticalFeed.map((feed, idx) => (
                <div key={idx} className="flex gap-3">
                  <span className="text-on-surface-variant/40">[{feed.timestamp}]</span>
                  <span className={getFeedTypeColor(feed.type)}>{feed.type}</span>
                  <span className="text-on-surface-variant">{feed.message}</span>
                </div>
              ))}
            </div>
            <div className="p-2 border-t border-outline-variant/30 bg-surface-container-low/50">
              <div className="flex justify-center items-center gap-4 py-2">
                <div className="w-1 h-1 bg-surface-tint rounded-full animate-ping"></div>
                <span className="text-label-caps font-label-caps text-surface-tint tracking-[0.3em]">PROCESSING DATA STREAM</span>
              </div>
            </div>
          </section>
        </div>
      </main>

      <AppBottomNav className="md:hidden" />
    </div>
  );
}
