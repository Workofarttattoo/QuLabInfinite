import { Navigation } from '../components/Navigation';
import { useSystemStatus, useLiveMetrics } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function StormIntelligence() {
  const { systems, loading } = useSystemStatus();
  const { metrics } = useLiveMetrics();

  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 px-[32px]">
          <EchoLabCommandInline className="mb-8" />
        <div className="max-w-[1440px] mx-auto">
          <div className="mb-12">
            <h1 className="text-[48px] leading-[56px] tracking-[-0.02em] font-bold text-[#7df4ff] mb-4">
              PRECISION STORM INTELLIGENCE
            </h1>
            <p className="text-[18px] leading-[28px] text-[#b9cacb]">
              Real-time operational intelligence and system-wide monitoring
            </p>
          </div>

          <div className="grid grid-cols-3 gap-6 mb-8">
            {!loading && systems.map((sys) => (
              <div key={sys.id} className={`glass-panel p-6 rounded-xl border-l-4 ${
                sys.status === 'operational' ? 'border-l-[#00dbe9] neon-glow-cyan' :
                sys.status === 'syncing' ? 'border-l-[#ddb7ff] neon-glow-purple' :
                'border-l-[#ffb4ab]'
              }`}>
                <div className="mb-4">
                  <div className="text-[10px] tracking-[0.15em] font-bold text-[#b9cacb] mb-2">REGION</div>
                  <div className="text-[20px] font-semibold text-[#dce4e5]">{sys.region}</div>
                </div>
                <div className="mb-4">
                  <div className="text-[10px] text-[#b9cacb] mb-1">Status</div>
                  <div className={`text-[16px] font-bold ${
                    sys.status === 'operational' ? 'text-[#00dbe9]' :
                    sys.status === 'syncing' ? 'text-[#ddb7ff]' :
                    'text-[#ffb4ab]'
                  }`}>
                    {sys.status.toUpperCase()}
                  </div>
                </div>
                {sys.uptime > 0 && (
                  <div className="mb-2">
                    <div className="text-[10px] text-[#b9cacb] mb-1">Uptime</div>
                    <div className="text-[24px] font-bold text-[#00dbe9]">{sys.uptime}%</div>
                  </div>
                )}
                {sys.latency && (
                  <div className="text-[12px] text-[#b9cacb] pt-3 border-t border-white/10">
                    Latency: <span className="text-[#ddb7ff]">{sys.latency}ms</span>
                  </div>
                )}
                {sys.node_count && (
                  <div className="text-[12px] text-[#b9cacb] mt-1">
                    Nodes: <span className="text-[#00dbe9]">{sys.node_count.toLocaleString()}</span>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="glass-panel p-8 rounded-xl neon-glow-cyan mb-8">
            <h3 className="text-[24px] font-semibold text-[#00dbe9] mb-6">Live System Metrics</h3>
            <div className="grid grid-cols-4 gap-6">
              <div>
                <div className="text-[12px] text-[#b9cacb] mb-2">Total Nodes</div>
                <div className="text-[32px] font-bold text-[#00dbe9]">
                  {systems.reduce((acc, s) => acc + (s.node_count || 0), 0).toLocaleString()}
                </div>
              </div>
              <div>
                <div className="text-[12px] text-[#b9cacb] mb-2">Avg Latency</div>
                <div className="text-[32px] font-bold text-[#ddb7ff]">
                  {systems.length > 0 && systems.some(s => s.latency)
                    ? (systems.reduce((acc, s) => acc + (s.latency || 0), 0) / systems.filter(s => s.latency).length).toFixed(2)
                    : '0'
                  }ms
                </div>
              </div>
              <div>
                <div className="text-[12px] text-[#b9cacb] mb-2">Operational</div>
                <div className="text-[32px] font-bold text-[#00dbe9]">
                  {systems.filter(s => s.status === 'operational').length}
                </div>
              </div>
              <div>
                <div className="text-[12px] text-[#b9cacb] mb-2">System Health</div>
                <div className="text-[32px] font-bold text-[#00dbe9]">
                  {systems.length > 0 && systems.some(s => s.uptime)
                    ? (systems.reduce((acc, s) => acc + (s.uptime || 0), 0) / systems.filter(s => s.uptime).length).toFixed(1)
                    : '0'
                  }%
                </div>
              </div>
            </div>
          </div>

          <div className="glass-panel p-6 rounded-xl">
            <h3 className="text-[24px] font-semibold text-[#dce4e5] mb-6">Recent Telemetry</h3>
            <div className="space-y-2">
              {metrics.slice(0, 8).map((metric, idx) => (
                <div key={`${metric.id}-${idx}`} className="flex items-center justify-between p-3 bg-[#192122] rounded-lg">
                  <span className="text-[14px] text-[#b9cacb]">{metric.metric_name.replace(/_/g, ' ').toUpperCase()}</span>
                  <div className="flex items-center gap-4">
                    <span className="text-[16px] font-semibold text-[#00dbe9]">{metric.metric_value}</span>
                    <span className="text-[12px] text-[#b9cacb]">{metric.metric_type}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
