import { Navigation } from '../components/Navigation';
import { useLabHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function SepsisLab() {
  const { health, loading } = useLabHealth('sepsis');

  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 px-[32px]">
          <EchoLabCommandInline className="mb-8" />
        <div className="max-w-[1440px] mx-auto">
          <h1 className="text-[48px] font-bold text-[#ffb4ab] mb-4">Sepsis Early Warning System</h1>
          <p className="text-[18px] text-[#b9cacb] mb-8">qSOFA, SOFA, NEWS2 scoring - Port 8004</p>
          <div className="glass-panel p-6 rounded-xl">
            <h3 className="text-[24px] font-semibold text-[#ffb4ab] mb-4">Lab Status</h3>
            {loading ? <div className="text-[#b9cacb]">Loading...</div> : health ? <div className="text-[#00dbe9]">✅ {health.status}</div> : <div className="text-[#ffb4ab]">❌ Offline</div>}
          </div>
        </div>
      </main>
    </div>
  );
}
