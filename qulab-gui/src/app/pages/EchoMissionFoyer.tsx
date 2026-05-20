import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router';
import { apiClient } from '../../lib/api-client';
import { executeEchoCommand, type EchoCommandResult } from '../../lib/echo-command';
import { AppBottomNav } from '../components/AppBottomNav';

interface Quest {
  id: string;
  title: string;
  description: string;
  priority: 'PRIORITY_HIGH' | 'IN_PROGRESS' | 'COMPLETED';
  progress?: string;
  agents: string[];
}

interface LabPulseMetric {
  name: string;
  value: number;
  color: string;
}

export function EchoMissionFoyer() {
  const navigate = useNavigate();
  const [quests, setQuests] = useState<Quest[]>([
    {
      id: 'quest_1',
      title: 'Integrate with Notion',
      description: 'Synchronizing neural logs with the external workspace. Mapping relational databases to bio-rhythms.',
      priority: 'PRIORITY_HIGH',
      progress: '65%_SYNCED',
      agents: ['E', 'N']
    },
    {
      id: 'quest_2',
      title: 'Optimize DNA Sequencer',
      description: 'Recalibrating fluorescence detectors for 4th generation nanopore arrays. Reducing noise-to-signal ratio.',
      priority: 'IN_PROGRESS',
      progress: 'CALIBRATING',
      agents: ['E', 'Q']
    }
  ]);

  const [systemTraining, setSystemTraining] = useState({
    active: true,
    progress: 82,
    elapsed: '4H 12M',
    target: 'echo_intel_map.v2'
  });

  const [labPulse, setLabPulse] = useState<LabPulseMetric[]>([
    { name: 'REAGENT_LEVELS', value: 92.4, color: 'surface-tint' },
    { name: 'ROBOT_EFFICIENCY', value: 78.1, color: 'secondary-fixed-dim' }
  ]);

  const [coreTemp, setCoreTemp] = useState(-196);
  const [latency, setLatency] = useState(0.02);
  const [commandDraft, setCommandDraft] = useState('');
  const [isExecuting, setIsExecuting] = useState(false);
  const [lastResult, setLastResult] = useState<EchoCommandResult | null>(null);

  useEffect(() => {
    // Fetch agent telemetry from backend
    const fetchTelemetry = async () => {
      try {
        const telemetry = await apiClient.getAgentTelemetry();
        // Update metrics based on telemetry
        if (telemetry.neural_load) {
          setLabPulse(prev => [
            { name: 'REAGENT_LEVELS', value: telemetry.neural_load.cognitive_overhead * 100, color: 'surface-tint' },
            { name: 'ROBOT_EFFICIENCY', value: (1 - telemetry.neural_load.lattice_stress) * 100, color: 'secondary-fixed-dim' }
          ]);
        }
      } catch (error) {
        console.error('Failed to fetch telemetry:', error);
      }
    };

    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 20000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Simulate training progress
    const interval = setInterval(() => {
      setSystemTraining(prev => ({
        ...prev,
        progress: Math.min(100, prev.progress + Math.random() * 2)
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const handleQuestSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const text = commandDraft.trim();
    if (!text || isExecuting) return;

    setIsExecuting(true);
    setLastResult(null);

    const newQuest: Quest = {
      id: `quest_${Date.now()}`,
      title: text,
      description: 'Executing via MCP…',
      priority: 'IN_PROGRESS',
      progress: 'RUNNING',
      agents: ['E'],
    };
    setQuests((prev) => [newQuest, ...prev]);
    setCommandDraft('');

    const outcome = await executeEchoCommand(text, {
      context: { pathname: '/echo-mission', labSlug: 'echo-mission', labName: 'Echo Mission Foyer' },
    });
    setLastResult(outcome);
    setQuests((prev) =>
      prev.map((q) =>
        q.id === newQuest.id
          ? {
              ...q,
              description: outcome.detail ?? outcome.summary,
              priority: outcome.ok ? 'IN_PROGRESS' : 'PRIORITY_HIGH',
              progress: outcome.ok ? 'ACK' : 'FAILED',
            }
          : q
      )
    );
    setIsExecuting(false);
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'PRIORITY_HIGH':
        return 'text-primary-fixed-dim bg-primary-container/10';
      case 'IN_PROGRESS':
        return 'text-secondary-fixed-dim bg-secondary-container/10';
      case 'COMPLETED':
        return 'text-on-surface-variant bg-surface-variant/10';
      default:
        return 'text-on-surface-variant bg-surface-variant/10';
    }
  };

  return (
    <div className="min-h-screen bg-background text-on-surface font-body-md selection:bg-surface-tint selection:text-on-primary">
      {/* Scanline overlay */}
      <div className="fixed inset-0 scanline-overlay pointer-events-none z-50" aria-hidden />

      {/* Top Navigation Bar */}
      <nav className="bg-background/80 backdrop-blur-xl text-surface-tint border-b border-outline-variant/30 flex justify-between items-center w-full px-margin-mobile md:px-margin-desktop py-unit fixed top-0 z-40">
        <div className="flex items-center space-x-2">
          <span className="material-symbols-outlined">terminal</span>
          <span className="font-headline-sm text-headline-sm font-bold text-surface-tint tracking-tighter">QULAB_INF_OS // V.1.0.4</span>
        </div>
        <div className="hidden md:flex space-x-6 items-center">
          <span className="font-label-caps text-label-caps text-primary-fixed-dim">[ ENCRYPTED ]</span>
          <button
            onClick={() => navigate('/system')}
            className="material-symbols-outlined cursor-pointer hover:bg-primary-container/10 p-1 transition-colors"
          >
            settings
          </button>
        </div>
      </nav>

      <main className="pt-20 pb-24 px-margin-mobile md:px-margin-desktop min-h-screen grid grid-cols-1 md:grid-cols-12 gap-gutter max-w-7xl mx-auto">
        {/* Left Column: Echo Persona & Live Pulse */}
        <div className="md:col-span-4 flex flex-col space-y-tile-gap">
          {/* Persona Card */}
          <div className="glass-panel p-gutter rounded-none relative overflow-hidden group">
            <div className="flex justify-between items-start mb-4">
              <span className="font-label-caps text-label-caps text-surface-tint opacity-70">SUBJECT // ECHO_01</span>
              <span className="h-2 w-2 bg-primary-fixed-dim rounded-full"></span>
            </div>
            <div className="relative aspect-square w-full bg-[#0a0e14] mb-4 overflow-hidden border border-outline-variant/30">
              <img
                className="w-full h-full object-cover opacity-75 contrast-125 saturate-50 mix-blend-luminosity group-hover:opacity-90 transition-all duration-500"
                src="https://lh3.googleusercontent.com/aida-public/AB6AXuCnJqJ28wXQKCiW0wQwQV75DA9azp2HsxUp5-absTcn4NeIdyqyxpXtjhqNFcxWw0ECUHYAU8nMDMiqAoRI2X1eHVJD1vTwCX23cSpE7Pr8Rca7jgbWa-8UvaFhgmJQp6jm_gSOthrT1aa_d7OrTWwzYmm3H7NQToZ3LSoBhpFC0r5M89mgjcMEx2HeEv8MtVlJvSgM7ewBm3u5II2In1kcjxSe7vaWRdiYptONlpkFL2udibrwTD11_3p4r6WLxQvO2MAgGXjarIY"
                alt="Echo - Cognitive Synthetic Research Assistant"
              />
              <div
                className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#131313] via-[#131313]/70 to-[#00dbe9]/10"
                aria-hidden
              />
            </div>
            <div className="space-y-2">
              <h2 className="font-headline-sm text-headline-sm text-on-surface">ECHO</h2>
              <p className="font-body-md text-body-md text-on-surface-variant leading-relaxed">
                Cognitive Synthetic Research Assistant
              </p>
              <div className="flex flex-wrap gap-2 mt-4">
                <span className="px-2 py-0.5 border border-primary-fixed-dim text-primary-fixed-dim font-label-caps text-[9px]">
                  BIO_ENGINEER
                </span>
                <span className="px-2 py-0.5 border border-outline-variant text-on-surface-variant font-label-caps text-[9px]">
                  LEVEL_MAX
                </span>
              </div>
            </div>
          </div>

          {/* Lab Pulse Visualization */}
          <div className="glass-panel p-gutter rounded-none flex-grow">
            <div className="flex justify-between items-center mb-6">
              <span className="font-label-caps text-label-caps text-surface-tint">LAB_PULSE_STREAM</span>
              <span className="text-secondary-fixed-dim text-label-caps font-bold">STABLE</span>
            </div>
            <div className="space-y-6">
              {labPulse.map((metric, idx) => (
                <div key={idx} className="space-y-2">
                  <div className="flex justify-between font-label-caps text-[10px] text-on-surface-variant">
                    <span>{metric.name}</span>
                    <span>{metric.value.toFixed(1)}%</span>
                  </div>
                  <div className="h-4 w-full bg-surface-container/50 border border-outline-variant/30 relative">
                    <div
                      className="h-full segment-bar opacity-80"
                      style={{
                        width: `${metric.value}%`,
                        backgroundImage: `repeating-linear-gradient(90deg, ${metric.color === 'surface-tint' ? '#00dbe9' : '#00e639'}, ${metric.color === 'surface-tint' ? '#00dbe9' : '#00e639'} 8px, transparent 8px, transparent 10px)`
                      }}
                    ></div>
                  </div>
                </div>
              ))}

              {/* Mini Telemetry */}
              <div className="grid grid-cols-2 gap-2 pt-4 border-t border-outline-variant/20">
                <div className="p-2 bg-surface-container-low">
                  <div className="font-label-caps text-[8px] text-on-surface-variant">TEMP_CORE</div>
                  <div className="font-data-display text-headline-sm text-surface-tint">{coreTemp}°C</div>
                </div>
                <div className="p-2 bg-surface-container-low">
                  <div className="font-label-caps text-[8px] text-on-surface-variant">LATENCY</div>
                  <div className="font-data-display text-headline-sm text-surface-tint">{latency}ms</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Intent Capture & Quests */}
        <div className="md:col-span-8 flex flex-col space-y-tile-gap">
          {/* Intent Capture */}
          <div className="glass-panel p-8 border-l-4 border-l-surface-tint">
            <div className="mb-4">
              <h1 className="font-headline-md text-headline-md text-on-surface tracking-tight mb-1">
                WELCOME BACK, OPERATIVE.
              </h1>
              <p className="font-body-md text-body-md text-on-surface-variant">
                System ready for new commands. Neural link verified.
              </p>
            </div>
            <form onSubmit={handleQuestSubmit} className="relative mt-8 group">
              <div className="absolute inset-y-0 left-4 flex items-center pointer-events-none">
                <span className="material-symbols-outlined text-surface-tint">chevron_right</span>
              </div>
              <input
                className="w-full bg-surface-container-low border border-outline-variant/50 px-12 py-5 font-body-md text-on-surface focus:ring-0 focus:border-surface-tint transition-all placeholder:text-on-surface-variant/30 text-lg disabled:opacity-60"
                placeholder="Whisper a quest to Echo..."
                type="text"
                value={commandDraft}
                onChange={(e) => setCommandDraft(e.target.value)}
                disabled={isExecuting}
                autoComplete="off"
              />
              <div className="absolute right-4 top-1/2 -translate-y-1/2">
                <button
                  type="submit"
                  disabled={isExecuting || !commandDraft.trim()}
                  className="bg-primary-container/10 border border-surface-tint px-4 py-2 font-label-caps text-surface-tint hover:bg-surface-tint hover:text-on-primary transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {isExecuting ? 'RUNNING…' : 'EXECUTE'}
                </button>
              </div>
            </form>
            {lastResult && (
              <div
                className={`mt-4 p-3 border text-sm font-body-md ${
                  lastResult.ok
                    ? 'border-secondary-fixed-dim/50 bg-secondary-container/10 text-on-surface'
                    : 'border-error/50 bg-error/10 text-error'
                }`}
                role="status"
              >
                <span className="font-label-caps text-[10px] block mb-1 text-surface-tint">
                  {lastResult.tool ?? 'ECHO'} // {lastResult.ok ? 'OK' : 'ERR'}
                </span>
                {lastResult.summary}
                {lastResult.detail ? (
                  <p className="mt-1 text-on-surface-variant text-xs break-words">{lastResult.detail}</p>
                ) : null}
              </div>
            )}
            <div className="mt-3 flex space-x-4">
              <span className="font-label-caps text-[9px] text-on-surface-variant/50">
                TRY: status · help · or any directive (logged via MCP)
              </span>
            </div>
          </div>

          {/* Active Quests Grid */}
          <div className="flex-grow">
            <div className="flex items-center space-x-2 mb-4">
              <span className="material-symbols-outlined text-surface-tint text-sm">assignment_late</span>
              <h3 className="font-label-caps text-label-caps text-surface-tint">ACTIVE_MISSIONS_LOG</h3>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-tile-gap">
              {/* Quest Cards */}
              {quests.map((quest) => (
                <div
                  key={quest.id}
                  className="glass-panel p-gutter hover:border-surface-tint/60 transition-all cursor-pointer group"
                  onClick={() => navigate(`/quest/${quest.id}`)}
                >
                  <div className="flex justify-between mb-4">
                    <span className={`font-label-caps text-[10px] px-2 py-0.5 ${getPriorityColor(quest.priority)}`}>
                      {quest.priority}
                    </span>
                    <span className="material-symbols-outlined text-surface-tint text-lg group-hover:rotate-45 transition-transform">
                      north_east
                    </span>
                  </div>
                  <h4 className="font-headline-sm text-headline-sm mb-2">{quest.title}</h4>
                  <p className="font-body-md text-body-md text-on-surface-variant mb-6 h-12 overflow-hidden">
                    {quest.description}
                  </p>
                  <div className="flex items-center justify-between border-t border-outline-variant/30 pt-4">
                    <div className="flex -space-x-2">
                      {quest.agents.map((agent, idx) => (
                        <div
                          key={idx}
                          className="w-6 h-6 rounded-full border border-background bg-surface-variant flex items-center justify-center text-[8px] font-bold"
                        >
                          {agent}
                        </div>
                      ))}
                    </div>
                    {quest.progress && (
                      <span className="font-label-caps text-[10px] text-on-surface">{quest.progress}</span>
                    )}
                  </div>
                </div>
              ))}

              {/* System Training Status */}
              {systemTraining.active && (
                <div className="glass-panel p-gutter bg-surface-tint/5 border-surface-tint/40 sm:col-span-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="p-3 bg-surface-tint text-on-primary">
                        <span className="material-symbols-outlined">psychology</span>
                      </div>
                      <div>
                        <h4 className="font-label-caps text-label-caps text-on-surface">SYSTEM_TRAINING_ACTIVE</h4>
                        <p className="font-body-md text-body-md text-surface-tint">
                          Processing {systemTraining.target}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-data-display text-headline-sm">{systemTraining.progress.toFixed(0)}%</div>
                      <div className="font-label-caps text-[9px] text-on-surface-variant">
                        ELAPSED: {systemTraining.elapsed}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <AppBottomNav />

      {/* Floating UI element for mission context */}
      <div className="fixed bottom-20 right-8 z-40 hidden md:block">
        <button
          onClick={() => {
            const input = document.querySelector('input[placeholder*="Whisper"]') as HTMLInputElement;
            input?.focus();
          }}
          className="h-14 w-14 bg-surface-tint text-on-primary flex items-center justify-center rounded-none shadow-lg border-glow-cyan hover:scale-95 transition-all"
        >
          <span className="material-symbols-outlined">add</span>
        </button>
      </div>
    </div>
  );
}
