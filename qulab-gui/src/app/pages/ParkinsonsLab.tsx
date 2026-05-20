import { Navigation } from '../components/Navigation';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function ParkinsonsLab() {
  const { health, loading } = useLabHealth('parkinsons');

  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 px-[32px]">
          <EchoLabCommandInline className="mb-8" />
        <div className="max-w-[1440px] mx-auto">
          <h1 className="text-[48px] leading-[56px] tracking-[-0.02em] font-bold text-[#ddb7ff] mb-4">
            Parkinson's Progression Predictor
          </h1>
          <p className="text-[18px] leading-[28px] text-[#b9cacb] mb-8">
            MDS-UPDRS, Hoehn & Yahr staging - Port 8002
          </p>

          <div className="glass-panel p-6 rounded-xl">
            <h3 className="text-[24px] font-semibold text-[#ddb7ff] mb-4">Lab Status</h3>
            {loading ? (
              <div className="text-[#b9cacb]">Loading...</div>
            ) : health ? (
              <div className="text-[#ddb7ff]">✅ {health.status}</div>
            ) : (
              <div className="text-[#ffb4ab]">❌ Offline</div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
