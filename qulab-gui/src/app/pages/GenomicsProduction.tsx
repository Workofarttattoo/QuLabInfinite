import { Navigation } from '../components/Navigation';
import { useGenomicsData } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

export function GenomicsProduction() {
  const { genomics, loading } = useGenomicsData();

  return (
    <div className="min-h-screen qulab-page-bg">
      <Navigation />
      <main className="relative pt-32 pb-20 px-[32px]">
          <EchoLabCommandInline className="mb-8" />
        <div className="max-w-[1440px] mx-auto">
          <div className="mb-12">
            <h1 className="text-[48px] leading-[56px] tracking-[-0.02em] font-bold text-[#7df4ff] mb-4">
              Refined Genomics Production Suite
            </h1>
            <p className="text-[18px] leading-[28px] text-[#b9cacb]">
              High-throughput genomic sequencing and production-grade analysis pipeline
            </p>
          </div>

          <div className="grid grid-cols-2 gap-6 mb-8">
            <div className="glass-panel p-8 rounded-xl neon-glow-cyan">
              <h3 className="text-[24px] font-semibold text-[#00dbe9] mb-4">Production Pipeline</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center pb-3 border-b border-white/10">
                  <span className="text-[#b9cacb]">Samples in Queue</span>
                  <span className="text-[24px] font-bold text-[#00dbe9]">{genomics.filter(g => g.status === 'processing').length}</span>
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-white/10">
                  <span className="text-[#b9cacb]">Completed Today</span>
                  <span className="text-[24px] font-bold text-[#ddb7ff]">{genomics.filter(g => g.status === 'completed').length}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#b9cacb]">Throughput Rate</span>
                  <span className="text-[24px] font-bold text-[#00dbe9]">98.5%</span>
                </div>
              </div>
            </div>

            <div className="glass-panel p-8 rounded-xl neon-glow-purple">
              <h3 className="text-[24px] font-semibold text-[#ddb7ff] mb-4">Quality Metrics</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center pb-3 border-b border-white/10">
                  <span className="text-[#b9cacb]">Avg Quality Score</span>
                  <span className="text-[24px] font-bold text-[#ddb7ff]">
                    {genomics.length > 0
                      ? (genomics.reduce((acc, g) => acc + (g.quality_score || 0), 0) / genomics.length).toFixed(1)
                      : '0'
                    }%
                  </span>
                </div>
                <div className="flex justify-between items-center pb-3 border-b border-white/10">
                  <span className="text-[#b9cacb]">Pass Rate</span>
                  <span className="text-[24px] font-bold text-[#00dbe9]">96.2%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[#b9cacb]">Error Rate</span>
                  <span className="text-[24px] font-bold text-[#ffb4ab]">0.8%</span>
                </div>
              </div>
            </div>
          </div>

          <div className="glass-panel p-6 rounded-xl">
            <h3 className="text-[24px] font-semibold text-[#dce4e5] mb-6">Recent Production Runs</h3>
            <div className="space-y-3">
              {!loading && genomics.slice(0, 5).map((sample) => (
                <div key={sample.id} className="flex items-center justify-between p-4 bg-[#192122] rounded-lg hover:bg-[#232b2c] transition-colors">
                  <div className="flex items-center gap-4">
                    <div className={`w-3 h-3 rounded-full ${sample.status === 'completed' ? 'bg-[#00dbe9]' : 'bg-[#ddb7ff]'} animate-pulse`}></div>
                    <div>
                      <div className="text-[16px] font-semibold text-[#dce4e5]">{sample.sample_id}</div>
                      <div className="text-[12px] text-[#b9cacb]">{sample.chromosome || 'N/A'}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-8">
                    <div>
                      <div className="text-[10px] text-[#b9cacb]">Expression</div>
                      <div className="text-[14px] font-semibold text-[#00dbe9]">{sample.gene_expression?.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[#b9cacb]">Quality</div>
                      <div className="text-[14px] font-semibold text-[#ddb7ff]">{sample.quality_score?.toFixed(1)}%</div>
                    </div>
                    <div className={`px-3 py-1 rounded-full ${sample.status === 'completed' ? 'bg-[#00f0ff]/20' : 'bg-[#6f00be]/20'}`}>
                      <span className={`text-[10px] tracking-[0.15em] font-bold ${sample.status === 'completed' ? 'text-[#00dbe9]' : 'text-[#ddb7ff]'}`}>
                        {sample.status.toUpperCase()}
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
