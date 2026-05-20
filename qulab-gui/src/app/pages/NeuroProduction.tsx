import { Navigation } from '../components/Navigation';
import { useNeuroData } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function NeuroProduction() {
  const { neuro, loading } = useNeuroData();

  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 px-[32px]">
          <EchoLabCommandInline className="mb-8" />
        <div className="max-w-[1440px] mx-auto">
          <div className="mb-12">
            <h1 className="text-[48px] leading-[56px] tracking-[-0.02em] font-bold text-[#ddb7ff] mb-4">
              Refined Neuro-Production Suite
            </h1>
            <p className="text-[18px] leading-[28px] text-[#b9cacb]">
              Production-grade neurological diagnostics and brain imaging analysis
            </p>
          </div>

          <div className="grid grid-cols-2 gap-6 mb-8">
            <div className="glass-panel p-8 rounded-xl neon-glow-purple">
              <h3 className="text-[24px] font-semibold text-[#ddb7ff] mb-4">Scan Pipeline</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center pb-3 border-b border-white/10">
                  <span className="text-[#b9cacb]">Scans in Progress</span>
                  <span className="text-[24px] font-bold text-[#ddb7ff]">{neuro.filter(n => n.status === 'processing').length}</span>
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-white/10">
                  <span className="text-[#b9cacb]">Completed Scans</span>
                  <span className="text-[24px] font-bold text-[#00dbe9]">{neuro.filter(n => n.status === 'completed').length}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#b9cacb]">Success Rate</span>
                  <span className="text-[24px] font-bold text-[#ddb7ff]">99.1%</span>
                </div>
              </div>
            </div>

            <div className="glass-panel p-8 rounded-xl neon-glow-cyan">
              <h3 className="text-[24px] font-semibold text-[#00dbe9] mb-4">Activity Metrics</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center pb-3 border-b border-white/10">
                  <span className="text-[#b9cacb]">Avg Activity Level</span>
                  <span className="text-[24px] font-bold text-[#00dbe9]">
                    {neuro.length > 0
                      ? (neuro.reduce((acc, n) => acc + (n.activity_level || 0), 0) / neuro.length).toFixed(1)
                      : '0'
                    }%
                  </span>
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-white/10">
                  <span className="text-[#b9cacb]">Signal Quality</span>
                  <span className="text-[24px] font-bold text-[#ddb7ff]">High</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#b9cacb]">Processing Time</span>
                  <span className="text-[24px] font-bold text-[#00dbe9]">2.4min</span>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-panel p-6 rounded-xl">
            <h3 className="text-[24px] font-semibold text-[#dce4e5] mb-6">Recent Diagnostic Scans</h3>
            <div className="space-y-3">
              {!loading && neuro.slice(0, 5).map((scan) => (
                <div key={scan.id} className="flex items-center justify-between p-4 bg-[#192122] rounded-lg hover:bg-[#232b2c] transition-colors">
                  <div className="flex items-center gap-4">
                    <div className={`w-3 h-3 rounded-full ${scan.status === 'completed' ? 'bg-[#ddb7ff]' : 'bg-[#00dbe9]'} animate-pulse`}></div>
                    <div>
                      <div className="text-[16px] font-semibold text-[#dce4e5]">{scan.patient_id}</div>
                      <div className="text-[12px] text-[#b9cacb]">{scan.brain_region?.replace(/_/g, ' ') || 'N/A'}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-8">
                    <div>
                      <div className="text-[10px] text-[#b9cacb]">Scan Type</div>
                      <div className="text-[14px] font-semibold text-[#ddb7ff]">{scan.scan_type}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[#b9cacb]">Activity</div>
                      <div className="text-[14px] font-semibold text-[#00dbe9]">{scan.activity_level?.toFixed(1)}%</div>
                    </div>
                    <div className={`px-3 py-1 rounded-full ${scan.status === 'completed' ? 'bg-[#6f00be]/20' : 'bg-[#00f0ff]/20'}`}>
                      <span className={`text-[10px] tracking-[0.15em] font-bold ${scan.status === 'completed' ? 'text-[#ddb7ff]' : 'text-[#00dbe9]'}`}>
                        {scan.status.toUpperCase()}
                      </span>
                    </div>
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
