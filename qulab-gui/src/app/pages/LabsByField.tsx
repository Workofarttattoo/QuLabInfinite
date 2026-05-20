import { useLabsHealth, useLabsConfig } from '../../lib/hooks';
import { getLabRoute } from '../../lib/lab-routes';
import { Link } from 'react-router';
import { useMemo } from 'react';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function LabsByField() {
  const { labsStatus, loading } = useLabsHealth();
  const { labs } = useLabsConfig();

  const labsByCategory = useMemo(() => {
    const labsArray = Object.entries(labs).map(([key, config]) => ({
      key,
      ...config,
      status: labsStatus[key],
    }));

    return {
      medical: labsArray.filter(lab => lab.category === 'medical'),
      materials: labsArray.filter(lab => lab.category === 'materials'),
      quantum: labsArray.filter(lab => lab.category === 'quantum'),
      bio: labsArray.filter(lab => lab.category === 'bio'),
      drug: labsArray.filter(lab => lab.category === 'drug'),
    };
  }, [labs, labsStatus]);

  const categoryConfig = {
    medical: {
      title: 'Medical Diagnostics',
      icon: 'medical_services',
      gradient: 'from-red-500/20 to-pink-500/20',
      accentColor: '#ef4444',
    },
    materials: {
      title: 'Materials Science',
      icon: 'science',
      gradient: 'from-blue-500/20 to-cyan-500/20',
      accentColor: '#06b6d4',
    },
    quantum: {
      title: 'Quantum Computing',
      icon: 'memory',
      gradient: 'from-purple-500/20 to-violet-500/20',
      accentColor: '#a855f7',
    },
    bio: {
      title: 'Biological Sciences',
      icon: 'biotech',
      gradient: 'from-green-500/20 to-emerald-500/20',
      accentColor: '#10b981',
    },
    drug: {
      title: 'Drug Discovery',
      icon: 'medication',
      gradient: 'from-orange-500/20 to-amber-500/20',
      accentColor: '#f59e0b',
    },
  };

  return (
    <div className="min-h-screen qulab-page-bg font-['Space_Grotesk'] text-foreground">
      {/* Header */}
      <header className="sticky top-0 z-50 glass-dark border-b border-white/5 p-4">
        <Link to="/" className="flex items-center gap-3">
          <div className="bg-[#137fec]/20 p-2 rounded-lg border border-[#137fec]/30">
            <span className="material-symbols-outlined text-[#137fec] text-2xl">rocket_launch</span>
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">QuLab Infinite</h1>
            <p className="text-[10px] uppercase tracking-[0.2em] text-[#137fec]/80 font-semibold">
              Research Fields
            </p>
          </div>
        </Link>
      </header>

            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="p-6 space-y-8">
        {Object.entries(categoryConfig).map(([categoryKey, config]) => {
          const categoryLabs = labsByCategory[categoryKey as keyof typeof labsByCategory];
          if (categoryLabs.length === 0) return null;

          const onlineCount = categoryLabs.filter(lab => lab.status?.healthy).length;

          return (
            <section key={categoryKey}>
              {/* Category Header */}
              <div className="flex items-center gap-4 mb-4">
                <div
                  className="p-3 rounded-xl glass"
                  style={{ borderColor: config.accentColor + '40' }}
                >
                  <span
                    className="material-symbols-outlined text-3xl"
                    style={{ color: config.accentColor }}
                  >
                    {config.icon}
                  </span>
                </div>
                <div className="flex-1">
                  <h2 className="text-2xl font-bold">{config.title}</h2>
                  <p className="text-sm text-slate-400">
                    {onlineCount} of {categoryLabs.length} labs online
                  </p>
                </div>
              </div>

              {/* Labs Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {categoryLabs.map((lab) => {
                  const isOnline = lab.status?.healthy;

                  return (
                    <Link
                      key={lab.key}
                      to={getLabRoute(lab.key)}
                      className="relative group"
                    >
                      <div className={`glass rounded-xl p-4 h-full transition-all duration-300 hover:scale-105 ${
                        isOnline ? 'hover:shadow-lg' : 'opacity-60'
                      }`}>
                        {/* Status Indicator */}
                        <div className="flex items-center justify-between mb-3">
                          <div className="flex items-center gap-2">
                            <span
                              className={`size-2 rounded-full ${
                                isOnline ? 'bg-green-400 animate-pulse' : 'bg-slate-600'
                              }`}
                            ></span>
                            <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-400">
                              {isOnline ? 'Online' : 'Offline'}
                            </span>
                          </div>
                          {lab.type === 'medical' && (
                            <span className="text-[9px] px-2 py-0.5 rounded-full bg-white/5 text-slate-400">
                              Port {lab.port}
                            </span>
                          )}
                        </div>

                        {/* Lab Name */}
                        <h3 className="text-base font-bold mb-2 leading-tight">
                          {lab.name}
                        </h3>

                        {/* Architecture */}
                        <div className="text-[11px] text-slate-400 space-y-1">
                          <div className="flex items-center gap-1.5">
                            <span className="material-symbols-outlined text-xs">api</span>
                            <span>{lab.type === 'unified' ? 'Unified API' : 'Microservice'}</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <span className="material-symbols-outlined text-xs">route</span>
                            <span className="font-mono">{lab.endpoint}</span>
                          </div>
                        </div>

                        {/* Hover Gradient */}
                        <div
                          className={`absolute inset-0 rounded-xl bg-gradient-to-br ${config.gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10`}
                        ></div>
                      </div>
                    </Link>
                  );
                })}
              </div>
            </section>
          );
        })}
      </main>
    </div>
  );
}
