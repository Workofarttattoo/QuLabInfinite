import { Navigation } from '../components/Navigation';
import { useLabHealth, useLabThresholds } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function AlzheimersLab() {
  const { health, loading: healthLoading } = useLabHealth('alzheimers');
  const { thresholds, loading: thresholdsLoading } = useLabThresholds('alzheimers');

  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 px-[32px]">
          <EchoLabCommandInline className="mb-8" />
        <div className="max-w-[1440px] mx-auto">
          <div className="mb-12">
            <h1 className="text-[48px] leading-[56px] tracking-[-0.02em] font-bold text-[#7df4ff] mb-4">
              Alzheimer's Early Detection Lab
            </h1>
            <p className="text-[18px] leading-[28px] text-[#b9cacb]">
              ATN biomarker classification (Amyloid/Tau/Neurodegeneration) - Port 8001
            </p>
          </div>

          <div className="grid grid-cols-2 gap-6 mb-8">
            <div className="glass-panel p-6 rounded-xl">
              <h3 className="text-[24px] font-semibold text-[#00dbe9] mb-4">Lab Status</h3>
              {healthLoading ? (
                <div className="text-[#b9cacb]">Loading...</div>
              ) : health ? (
                <div className="text-[#00dbe9]">✅ {health.status}</div>
              ) : (
                <div className="text-[#ffb4ab]">❌ Offline</div>
              )}
            </div>

            <div className="glass-panel p-6 rounded-xl">
              <h3 className="text-[24px] font-semibold text-[#ddb7ff] mb-4">Clinical Standards</h3>
              <p className="text-[#b9cacb] text-[14px]">NIA-AA research framework (Jack et al., 2018)</p>
            </div>
          </div>

          {!thresholdsLoading && thresholds && (
            <div className="glass-panel p-6 rounded-xl">
              <h3 className="text-[24px] font-semibold text-[#dce4e5] mb-6">Clinical Thresholds</h3>
              <pre className="text-[#00dbe9] text-[14px] overflow-auto">
                {JSON.stringify(thresholds, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
