import { Navigation } from '../components/Navigation';
import { useLabsHealth, useLabsConfig } from '../../lib/hooks';
import { getLabRoute } from '../../lib/lab-routes';
import { Link } from 'react-router';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function LiveLabWall() {
  const { labsStatus, loading } = useLabsHealth();
  const { labs } = useLabsConfig();

  const labsArray = Object.entries(labs).map(([key, config]) => ({
    key,
    ...config,
    status: labsStatus[key],
  }));

  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 px-[32px]">
          <EchoLabCommandInline className="mb-8" />
        <div className="max-w-[1440px] mx-auto">
          <div className="mb-12">
            <h1 className="text-[48px] leading-[56px] tracking-[-0.02em] font-bold text-[#7df4ff] mb-4">
              QuLab Infinite Live Lab Wall
            </h1>
            <p className="text-[18px] leading-[28px] text-[#b9cacb]">
              Real-time visualization of all computational R&D operations - Materials science, quantum chemistry, digital twins, and diagnostic intelligence
            </p>
          </div>

          <div className="mb-8">
            <h2 className="text-[24px] font-semibold text-[#ddb7ff] mb-4">Core R&D Labs - Unified API (Port 8000)</h2>
            <div className="grid grid-cols-3 gap-6 mb-12">
              {['materials', 'quantum', 'chemistry'].map(labKey => {
                const lab = labsArray.find(l => l.key === labKey);
                if (!lab) return null;
                const isOnline = lab.status?.healthy ?? false;
                const colorClass = isOnline ? 'text-[#00dbe9]' : 'text-[#ffb4ab]';
                const bgClass = isOnline ? 'bg-[#00f0ff]/20' : 'bg-[#ffb4ab]/20';

                return (
                  <Link key={lab.key} to={getLabRoute(lab.key)} className="block">
                    <div className="glass-panel p-6 rounded-xl border-white/10 hover:border-[#ddb7ff]/60 neon-glow-purple transition-all h-full">
                      <div className="flex justify-between items-start mb-4">
                        <h3 className="text-[20px] font-semibold text-[#ddb7ff]">{lab.name}</h3>
                        <div className={`px-3 py-1 rounded-full ${bgClass}`}>
                          <span className={`text-[10px] tracking-[0.15em] font-bold ${colorClass}`}>
                            UNIFIED API
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 mb-4">
                        <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-[#00dbe9]' : 'bg-[#ffb4ab]'} animate-pulse`}></span>
                        <span className={`text-[14px] ${colorClass}`}>
                          {isOnline ? 'ONLINE' : 'OFFLINE'}
                        </span>
                      </div>
                      <div className="mt-4 pt-4 border-t border-white/10">
                        <span className="text-[12px] text-[#ddb7ff] hover:underline">Launch Lab →</span>
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>

          <div className="mb-4">
            <h2 className="text-[20px] font-semibold text-[#b9cacb] mb-4">Medical Diagnostic Labs (Ports 8001-8010)</h2>
          </div>
          <div className="grid grid-cols-2 gap-6 mb-12">
            {labsArray.filter(lab => !['materials', 'quantum', 'chemistry', 'oncology', 'drug', 'genomics', 'immune', 'metabolic'].includes(lab.key)).map(lab => {
              const isOnline = lab.status?.healthy ?? false;
              const colorClass = isOnline ? 'text-[#00dbe9]' : 'text-[#ffb4ab]';
              const bgClass = isOnline ? 'bg-[#00f0ff]/20' : 'bg-[#ffb4ab]/20';

              return (
                <Link key={lab.key} to={getLabRoute(lab.key)} className="block">
                  <div className="glass-panel p-6 rounded-xl border-white/10 hover:border-[#00dbe9]/40 transition-all h-full">
                    <div className="flex justify-between items-start mb-4">
                      <h3 className="text-[18px] font-semibold text-[#dce4e5]">{lab.name}</h3>
                      <div className={`px-3 py-1 rounded-full ${bgClass}`}>
                        <span className={`text-[10px] tracking-[0.15em] font-bold ${colorClass}`}>
                          PORT {lab.port}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 mb-4">
                      <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-[#00dbe9]' : 'bg-[#ffb4ab]'} animate-pulse`}></span>
                      <span className={`text-[12px] ${colorClass}`}>
                        {isOnline ? 'ONLINE' : 'OFFLINE'}
                      </span>
                    </div>
                    <div className="mt-4 pt-4 border-t border-white/10">
                      <span className="text-[12px] text-[#00dbe9] hover:underline">View Lab →</span>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>

          <div className="glass-panel p-6 rounded-xl neon-glow-cyan">
            <div className="flex justify-between items-center">
              <div>
                <h3 className="text-[24px] font-semibold text-[#00dbe9] mb-2">System Status</h3>
                <p className="text-[14px] text-[#b9cacb]">R&D computational labs - Materials, chemistry, simulation & diagnostics</p>
              </div>
              <div className="flex gap-8">
                <div className="text-center">
                  <div className="text-[32px] font-bold text-[#00dbe9]">
                    {Object.values(labsStatus).filter(s => s.healthy).length}
                  </div>
                  <div className="text-[12px] text-[#b9cacb]">Labs Online</div>
                </div>
                <div className="text-center">
                  <div className="text-[32px] font-bold text-[#ddb7ff]">{labsArray.length}</div>
                  <div className="text-[12px] text-[#b9cacb]">Total Labs</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
