import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { LABS } from '../../lib/api-client';
import { useLabsHealth } from '../../lib/hooks';
import { getLabRouteByPort } from '../../lib/app-nav';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';
import { AppBottomNav } from '../components/AppBottomNav';

interface LabTile {
  port: number;
  name: string;
  status: 'ACTIVE' | 'OFFLINE' | 'DEGRADED';
  icon: string;
  successMetric: number;
  confidence?: number;
  visualization: 'ellipse' | 'cloud' | 'grid' | 'pulse' | 'gradient' | 'lines' | 'stream';
}

export function MedicalDirectoryUnified() {
  const navigate = useNavigate();
  const { labsStatus, loading } = useLabsHealth();
  const [configuredLabs, setConfiguredLabs] = useState(20);
  const [reachableNodes, setReachableNodes] = useState(10);
  const [networkStatus, setNetworkStatus] = useState<'STABLE' | 'DEGRADED' | 'OFFLINE'>('STABLE');
  const [syncPercentage, setSyncPercentage] = useState(0.02);

  const [labTiles, setLabTiles] = useState<LabTile[]>([
    { port: 8001, name: 'BIO_SYNTH_01', status: 'ACTIVE', icon: 'science', successMetric: 94, confidence: 98.2, visualization: 'ellipse' },
    { port: 8002, name: 'NEURAL_INF_X', status: 'ACTIVE', icon: 'psychology', successMetric: 89, visualization: 'cloud' },
    { port: 8003, name: 'GENOM_MAP_04', status: 'ACTIVE', icon: 'genetics', successMetric: 91, visualization: 'lines' },
    { port: 8004, name: 'PATH_SENSE_A', status: 'ACTIVE', icon: 'coronavirus', successMetric: 76, visualization: 'ellipse' },
    { port: 8005, name: 'RAD_TREAT_IX', status: 'ACTIVE', icon: 'radiology', successMetric: 96, visualization: 'cloud' },
    { port: 8006, name: 'IMMUNO_SIM', status: 'ACTIVE', icon: 'biotech', successMetric: 93, visualization: 'grid' },
    { port: 8007, name: 'CARDIAC_DSP', status: 'ACTIVE', icon: 'monitor_heart', successMetric: 98, visualization: 'gradient' },
    { port: 8008, name: 'RENAL_FILT_0', status: 'ACTIVE', icon: 'water_drop', successMetric: 90, visualization: 'lines' },
    { port: 8009, name: 'TRAUMA_CORE', status: 'ACTIVE', icon: 'emergency', successMetric: 92, visualization: 'pulse' },
    { port: 8010, name: 'ENDO_FLOW_0', status: 'ACTIVE', icon: 'stream', successMetric: 95, visualization: 'stream' }
  ]);

  const [systemLog, setSystemLog] = useState<Array<{ time: string; type: string; message: string; status?: string }>>([
    { time: '08:44:12', type: 'INFO', message: 'INITIALIZING_LAB_HANDSHAKE: PORT 8001...', status: 'OK' },
    { time: '08:44:13', type: 'INFO', message: 'BUFFERING_PROBABILITY_CLOUDS: NODE_UNIFIED_01...', status: '99% CONFIDENCE' },
    { time: '08:44:15', type: 'WARNING', message: 'WARNING: LATENCY_SPIKE ON PORT 8004... ATTEMPTING_RECONNECT...' }
  ]);

  useEffect(() => {
    // Sync lab statuses with backend
    if (!loading && labsStatus) {
      const totalLabs = Object.keys(labsStatus).length;
      const healthyLabs = Object.values(labsStatus).filter(s => s.healthy).length;

      setConfiguredLabs(totalLabs);
      setReachableNodes(healthyLabs);
      setNetworkStatus(healthyLabs > totalLabs * 0.7 ? 'STABLE' : healthyLabs > 0 ? 'DEGRADED' : 'OFFLINE');

      // Update lab tiles based on actual status
      setLabTiles(prev =>
        prev.map(tile => {
          const labKey = Object.entries(LABS).find(([, config]) => config.port === tile.port)?.[0];

          if (labKey && labsStatus[labKey]) {
            return {
              ...tile,
              status: labsStatus[labKey].healthy ? 'ACTIVE' : 'OFFLINE'
            };
          }
          return tile;
        })
      );
    }
  }, [labsStatus, loading]);

  useEffect(() => {
    // Simulate system log updates
    const interval = setInterval(() => {
      const now = new Date();
      const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;

      const logMessages = [
        { type: 'INFO', message: 'NEURAL_BRIDGE_SYNC: Uplink validated...', status: 'OK' },
        { type: 'INFO', message: 'BATCH_PROCESSING: Cycle complete...', status: 'COMPLETE' },
        { type: 'WARNING', message: 'CACHE_THRESHOLD: Memory optimization triggered...' }
      ];

      const randomLog = logMessages[Math.floor(Math.random() * logMessages.length)];

      setSystemLog(prev => [
        ...prev.slice(-2),
        { time: timeStr, ...randomLog }
      ]);
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  const handleTryCommand = (_labName: string, port: number) => {
    navigate(getLabRouteByPort(port));
  };

  const renderVisualization = (type: string, icon: string, successMetric: number) => {
    switch (type) {
      case 'ellipse':
        return (
          <div className="relative h-20 mb-4 overflow-hidden border border-outline-variant/20">
            <div className={`error-ellipse absolute inset-2 ${successMetric < 80 ? 'border-error/30' : ''}`}></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className={`material-symbols-outlined ${successMetric < 80 ? 'text-error' : 'text-surface-tint'} opacity-30 text-4xl`}>{icon}</span>
            </div>
          </div>
        );
      case 'cloud':
        return (
          <div className="relative h-20 mb-4 overflow-hidden border border-outline-variant/20 bg-surface-container-lowest/50">
            <div className="prob-cloud absolute inset-0"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="material-symbols-outlined text-surface-tint opacity-30 text-4xl">{icon}</span>
            </div>
          </div>
        );
      case 'grid':
        return (
          <div className="h-20 bg-surface-container-lowest/30 border border-outline-variant/20 p-2 overflow-hidden flex flex-wrap gap-1 mb-4">
            {[...Array(18)].map((_, i) => (
              <div key={i} className="w-1 h-1 bg-surface-tint/40"></div>
            ))}
          </div>
        );
      case 'gradient':
        return (
          <div className="h-20 bg-surface-container-lowest/30 border border-outline-variant/20 relative mb-4">
            <div className="absolute bottom-0 left-0 w-full h-full bg-gradient-to-t from-surface-tint/10 to-transparent"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="material-symbols-outlined text-surface-tint opacity-30 text-4xl">{icon}</span>
            </div>
          </div>
        );
      case 'lines':
        return (
          <div className="h-20 bg-surface-container-lowest/30 border border-outline-variant/20 p-4 mb-4">
            <div className="w-full h-1 bg-surface-variant mb-1"></div>
            <div className="w-3/4 h-1 bg-surface-variant mb-1"></div>
            <div className="w-full h-1 bg-surface-tint/50"></div>
          </div>
        );
      case 'pulse':
        return (
          <div className="h-20 bg-surface-container-lowest/30 border border-outline-variant/20 flex items-center justify-center mb-4">
            <div className="w-12 h-12 rounded-full border-2 border-surface-tint/30 animate-pulse"></div>
          </div>
        );
      case 'stream':
        return (
          <div className="h-20 bg-surface-container-lowest/30 border border-outline-variant/20 flex items-center justify-center italic text-xs text-on-surface-variant/40 mb-4">
            REAL_TIME_STREAM_LIVE
          </div>
        );
      default:
        return (
          <div className="relative h-20 mb-4 overflow-hidden border border-outline-variant/20">
            <div className="absolute top-1/2 left-0 w-full h-[1px] bg-surface-tint/20"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="material-symbols-outlined text-surface-tint opacity-30 text-4xl">{icon}</span>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-background text-on-surface font-body-md selection:bg-primary-container selection:text-on-primary-container">
      {/* Top App Bar */}
      <header className="flex justify-between items-center w-full px-margin-mobile md:px-margin-desktop py-unit border-b border-outline-variant/50 bg-background/80 backdrop-blur-xl fixed top-0 z-50">
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-surface-tint" style={{ fontVariationSettings: '"FILL" 1' }}>terminal</span>
          <h1 className="font-headline-sm text-headline-sm font-bold text-surface-tint tracking-tighter">QULAB_INF_OS // V.1.0.4</h1>
        </div>
        <div className="hidden md:flex gap-gutter items-center">
          <span className="font-data-display text-data-display text-surface-tint">EST. SYNC: {syncPercentage.toFixed(1)}ms</span>
          <button className="px-4 py-2 border border-surface-tint text-surface-tint font-label-caps hover:bg-primary-container/10 transition-colors uppercase">
            [ENCRYPTED]
          </button>
        </div>
      </header>

            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="min-h-screen pt-24 pb-20 px-margin-mobile md:px-margin-desktop max-w-[1600px] mx-auto">
        {/* Medical Profile Header Section */}
        <section className="grid grid-cols-1 md:grid-cols-12 gap-tile-gap mb-gutter">
          <div className="md:col-span-8 glass-panel p-gutter flex flex-col md:flex-row justify-between items-start md:items-center gap-gutter relative overflow-hidden">
            <div className="prob-cloud absolute inset-0 -z-10"></div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="font-label-caps text-on-surface-variant">DIRECTORY_NODE:</span>
                <span className="text-surface-tint font-bold">MED_DIR_UNIFIED_01</span>
              </div>
              <h2 className="font-headline-md text-headline-md text-on-surface uppercase tracking-tight">System Diagnostic Hub</h2>
            </div>
            <div className="flex gap-4">
              <div className="flex flex-col items-end">
                <span className="font-label-caps text-on-surface-variant">CONFIGURED_LABS</span>
                <span className="font-data-display text-data-display text-surface-tint">{configuredLabs}</span>
              </div>
              <div className="w-[1px] h-12 bg-outline-variant/30"></div>
              <div className="flex flex-col items-end">
                <span className="font-label-caps text-on-surface-variant">REACHABLE_NODES</span>
                <span className="font-data-display text-data-display text-secondary-fixed-dim">{reachableNodes}</span>
              </div>
            </div>
          </div>
          <div className="md:col-span-4 glass-panel p-gutter border-l-2 border-surface-tint flex flex-col justify-center">
            <span className="font-label-caps text-on-surface-variant mb-2">NETWORK_STATUS</span>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${networkStatus === 'STABLE' ? 'bg-secondary-fixed-dim shadow-[0_0_8px_rgba(0,230,57,0.6)]' : 'bg-error'}`}></div>
              <span className={`font-bold ${networkStatus === 'STABLE' ? 'text-secondary-fixed-dim' : 'text-error'}`}>
                {networkStatus} // UPLINK_{networkStatus === 'STABLE' ? 'LOCKED' : 'UNSTABLE'}
              </span>
            </div>
            <div className="mt-4 h-2 w-full bg-surface-variant overflow-hidden">
              <div className="h-full bg-secondary-fixed-dim segmented-progress" style={{ width: `${(reachableNodes / configuredLabs) * 100}%` }}></div>
            </div>
          </div>
        </section>

        {/* Main Production Grid (Bento Style) */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-5 gap-tile-gap">
          {labTiles.map((lab, idx) => (
            <div
              key={idx}
              className="glass-panel p-4 flex flex-col justify-between min-h-[220px] active-cyan-glow group hover:bg-surface-variant/20 transition-all cursor-pointer"
              onClick={() => handleTryCommand(lab.name, lab.port)}
            >
              <div className="flex justify-between items-start mb-4">
                <div className="flex flex-col">
                  <span className="font-label-caps text-on-surface-variant">PORT_{lab.port}</span>
                  <span className="font-headline-sm text-headline-sm text-on-surface">{lab.name}</span>
                </div>
                <div className={`px-2 py-1 ${lab.status === 'ACTIVE' ? 'bg-surface-tint/20 border-surface-tint text-surface-tint' : 'bg-error/20 border-error text-error'} border font-label-caps text-[10px]`}>
                  {lab.status}
                </div>
              </div>
              {renderVisualization(lab.visualization, lab.icon, lab.successMetric)}
              <div className="flex justify-between items-end">
                <div>
                  <span className="font-label-caps text-on-surface-variant block mb-1">SUCCESS_METRIC</span>
                  <span className={`font-data-display text-data-display ${lab.successMetric < 80 ? 'text-error' : 'text-secondary-fixed-dim'}`}>
                    {lab.successMetric}% <span className="text-xs">KR</span>
                  </span>
                </div>
                <button className="font-body-md text-surface-tint hover:underline uppercase tracking-tighter">
                  &gt; TRY_CMD
                </button>
              </div>
            </div>
          ))}
        </section>

        {/* Command Console Interface (Tactile Detail) */}
        <section className="mt-gutter glass-panel p-gutter">
          <div className="flex items-center gap-4 border-b border-outline-variant/20 pb-4 mb-4">
            <span className="material-symbols-outlined text-surface-tint">terminal</span>
            <span className="font-label-caps">SYSTEM_LOG_OVERRIDE</span>
          </div>
          <div className="space-y-2 font-body-md text-xs text-on-surface-variant/80">
            {systemLog.map((log, idx) => (
              <p key={idx}>
                <span className={log.type === 'WARNING' ? 'text-error' : 'text-surface-tint'}>
                  [{log.time}]
                </span> {log.message}{' '}
                {log.status && <span className="text-secondary-fixed-dim">{log.status}</span>}
              </p>
            ))}
            <div className="flex items-center gap-2 mt-4 bg-surface-container/50 p-2 border border-outline-variant/30">
              <span className="text-surface-tint font-bold">root@qulab_os:~$</span>
              <input
                className="bg-transparent border-none outline-none focus:ring-0 p-0 w-full text-on-surface"
                type="text"
                defaultValue="deploy --target all --force"
                autoFocus
              />
              <span className="w-2 h-4 bg-surface-tint animate-pulse"></span>
            </div>
          </div>
        </section>
      </main>

      <AppBottomNav />
    </div>
  );
}
