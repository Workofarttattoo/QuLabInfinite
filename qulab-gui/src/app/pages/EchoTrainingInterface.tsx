import { useState } from 'react';
import { useNavigate } from 'react-router';
import { useLabsHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

interface TrainingDataset {
  id: string;
  name: string;
  size: string;
  status: 'active' | 'queued' | 'processing';
  samples: number;
}

interface TrainingLog {
  timestamp: string;
  epoch: number;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
}

interface ModelVersion {
  id: string;
  version: string;
  accuracy: number;
  trainedAt: string;
}

const EchoTrainingInterface = () => {
  const navigate = useNavigate();
  const { labsStatus, loading } = useLabsHealth();
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<string>('v2.4.1');
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [temperature, setTemperature] = useState<number>(0.7);
  const [topP, setTopP] = useState<number>(0.9);
  const [maxTokens, setMaxTokens] = useState<number>(2048);
  const [currentEpoch] = useState<number>(127);
  const [totalEpochs] = useState<number>(500);

  const [datasets] = useState<TrainingDataset[]>([
    { id: 'ds-001', name: 'NEURAL_CORPUS_PRIMARY', size: '42.3 GB', status: 'active', samples: 1847392 },
    { id: 'ds-002', name: 'SYNTHESIS_DATASET_ALPHA', size: '18.7 GB', status: 'processing', samples: 892041 },
    { id: 'ds-003', name: 'QUANTUM_BRIDGE_SEQUENCES', size: '9.2 GB', status: 'queued', samples: 421089 },
    { id: 'ds-004', name: 'TEMPORAL_REASONING_SET', size: '31.5 GB', status: 'active', samples: 1523876 },
  ]);

  const [trainingLogs] = useState<TrainingLog[]>([
    { timestamp: '2026-05-18T14:28:41Z', epoch: 127, message: 'Gradient descent converging optimally.', type: 'success' },
    { timestamp: '2026-05-18T14:28:33Z', epoch: 127, message: 'Validation batch #894 processed.', type: 'info' },
    { timestamp: '2026-05-18T14:28:20Z', epoch: 126, message: 'Loss spike detected - auto-adjusting learning rate.', type: 'warning' },
    { timestamp: '2026-05-18T14:28:12Z', epoch: 126, message: 'Epoch 126 completed. Accuracy: 94.82%.', type: 'success' },
    { timestamp: '2026-05-18T14:28:04Z', epoch: 126, message: 'Memory allocation exceeded threshold.', type: 'error' },
    { timestamp: '2026-05-18T14:27:51Z', epoch: 125, message: 'Checkpoint saved to quantum storage.', type: 'info' },
  ]);

  const [modelVersions] = useState<ModelVersion[]>([
    { id: 'mv-001', version: 'v2.4.1', accuracy: 94.82, trainedAt: '2026-05-18T09:15:00Z' },
    { id: 'mv-002', version: 'v2.4.0', accuracy: 93.67, trainedAt: '2026-05-17T22:30:00Z' },
    { id: 'mv-003', version: 'v2.3.9', accuracy: 92.44, trainedAt: '2026-05-16T18:45:00Z' },
    { id: 'mv-004', version: 'v2.3.8', accuracy: 91.28, trainedAt: '2026-05-15T11:20:00Z' },
  ]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-secondary-fixed-dim';
      case 'processing': return 'text-surface-tint';
      case 'queued': return 'text-on-surface-variant';
      default: return 'text-on-surface-variant';
    }
  };

  const getLogColor = (type: string) => {
    switch (type) {
      case 'success': return 'text-secondary-fixed-dim';
      case 'error': return 'text-error';
      case 'warning': return 'text-on-surface-variant';
      default: return 'text-surface-tint';
    }
  };

  const epochProgress = (currentEpoch / totalEpochs) * 100;

  const handleTrainingToggle = () => {
    setIsTraining(!isTraining);
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000 / 60);
    if (diff < 1) return 'just now';
    if (diff < 60) return `${diff}m ago`;
    return date.toLocaleTimeString();
  };

  return (
    <div className="min-h-screen bg-background text-on-surface font-body-md overflow-hidden selection:bg-surface-tint selection:text-on-primary">
      {/* Scanline Overlay */}
      <div className="fixed inset-0 pointer-events-none opacity-20 z-50"
           style={{
             background: 'linear-gradient(to bottom, transparent 50%, rgba(0, 219, 233, 0.03) 50%)',
             backgroundSize: '100% 4px'
           }}></div>

      {/* Top App Bar */}
      <header className="flex justify-between items-center w-full px-margin-mobile md:px-margin-desktop py-unit border-b border-outline-variant/50 bg-background/80 backdrop-blur-xl fixed top-0 z-40">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/')} className="material-symbols-outlined text-surface-tint hover:text-secondary-fixed-dim transition-colors">
            arrow_back
          </button>
          <span className="material-symbols-outlined text-surface-tint">model_training</span>
          <h1 className="font-headline-sm text-headline-sm font-bold text-surface-tint tracking-tighter">
            ECHO_TRAINING_INTERFACE // V.2.4.1
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-4 text-label-caps font-label-caps">
            <span className="text-secondary-fixed-dim">EPOCH: {currentEpoch}/{totalEpochs}</span>
            <span className="text-on-surface-variant/40">|</span>
            <span className="text-on-surface-variant">ACCURACY: 94.82%</span>
          </div>
          <button className="px-3 py-1 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors">
            [NEURAL_SYNC]
          </button>
        </div>
      </header>

      {/* Main Content Canvas */}
            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-20 pb-20 md:pb-8 min-h-screen">
        <div className="p-gutter grid grid-cols-12 gap-tile-gap">

          {/* Model Performance Metrics Dashboard */}
          <section className="col-span-12 lg:col-span-8"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex justify-between items-center">
              <div className="flex items-center gap-2">
                <span className="text-label-caps font-label-caps text-surface-tint">
                  PERFORMANCE_METRICS::REAL_TIME
                </span>
              </div>
              <div className="flex gap-4">
                <span className="text-label-caps font-label-caps text-secondary-fixed-dim flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-secondary-fixed-dim"></span> TRAINING
                </span>
                <span className="text-label-caps font-label-caps text-surface-tint flex items-center gap-1">
                  <span className="w-1.5 h-1.5 bg-surface-tint"></span> VALIDATION
                </span>
              </div>
            </div>
            <div className="p-6">
              {/* Mock Loss Function Graph */}
              <div className="relative h-64 bg-surface-container-lowest/50 border border-outline-variant/30 p-4">
                <div className="absolute top-4 left-4">
                  <span className="text-label-caps font-label-caps text-on-surface-variant">LOSS_FUNCTION</span>
                </div>
                {/* Mock graph visualization */}
                <svg className="w-full h-full" viewBox="0 0 400 200" preserveAspectRatio="none">
                  <polyline
                    fill="none"
                    stroke="#00dbe9"
                    strokeWidth="2"
                    points="0,180 40,160 80,140 120,110 160,90 200,75 240,65 280,58 320,54 360,52 400,51"
                    opacity="0.8"
                  />
                  <polyline
                    fill="none"
                    stroke="#00e639"
                    strokeWidth="2"
                    points="0,190 40,175 80,155 120,125 160,105 200,88 240,76 280,68 320,62 360,58 400,56"
                    opacity="0.6"
                  />
                </svg>
                <div className="absolute bottom-4 right-4 text-right">
                  <div className="font-data-display text-data-display text-surface-tint">
                    LOSS: 0.0428
                  </div>
                  <div className="font-label-caps text-label-caps text-on-surface-variant">
                    GRADIENT_NORM: 0.0089
                  </div>
                </div>
              </div>

              {/* Metric Cards */}
              <div className="grid grid-cols-3 gap-4 mt-6">
                <div className="p-4 border border-outline-variant/30 bg-surface-container-lowest/30">
                  <span className="text-label-caps font-label-caps text-on-surface-variant block mb-2">
                    TRAIN_ACCURACY
                  </span>
                  <div className="font-data-display text-data-display text-secondary-fixed-dim">94.82%</div>
                </div>
                <div className="p-4 border border-outline-variant/30 bg-surface-container-lowest/30">
                  <span className="text-label-caps font-label-caps text-on-surface-variant block mb-2">
                    VAL_ACCURACY
                  </span>
                  <div className="font-data-display text-data-display text-surface-tint">93.67%</div>
                </div>
                <div className="p-4 border border-outline-variant/30 bg-surface-container-lowest/30">
                  <span className="text-label-caps font-label-caps text-on-surface-variant block mb-2">
                    F1_SCORE
                  </span>
                  <div className="font-data-display text-data-display text-on-surface">0.9421</div>
                </div>
              </div>
            </div>
          </section>

          {/* Training Controls */}
          <div className="col-span-12 lg:col-span-4 flex flex-col gap-tile-gap">
            {/* Current Epoch Progress */}
            <article className="p-4 flex flex-col justify-between"
                     style={{
                       background: 'rgba(13, 18, 18, 0.7)',
                       backdropFilter: 'blur(12px)',
                       border: '0.5px solid rgba(0, 219, 233, 0.6)',
                       boxShadow: '0 0 10px rgba(0, 219, 233, 0.2)'
                     }}>
              <div className="flex justify-between items-start">
                <span className="font-label-caps text-label-caps text-on-surface-variant">
                  EPOCH_PROGRESS
                </span>
                <span className="material-symbols-outlined text-surface-tint">
                  {isTraining ? 'play_circle' : 'pause_circle'}
                </span>
              </div>
              <div className="mt-4">
                <div className="font-data-display text-data-display text-surface-tint">
                  {currentEpoch} / {totalEpochs}
                </div>
                <div className="text-label-caps font-label-caps text-on-surface-variant mt-1">
                  {epochProgress.toFixed(1)}% COMPLETE
                </div>
                <div className="w-full h-2 bg-outline-variant/30 mt-3">
                  <div
                    className="h-full bg-surface-tint transition-all duration-300"
                    style={{ width: `${epochProgress}%` }}
                  ></div>
                </div>
              </div>
            </article>

            {/* Model Version Selector */}
            <article className="p-4 flex flex-col"
                     style={{
                       background: 'rgba(13, 18, 18, 0.7)',
                       backdropFilter: 'blur(12px)',
                       border: '0.5px solid rgba(185, 202, 203, 0.15)'
                     }}>
              <div className="flex justify-between items-start mb-3">
                <span className="font-label-caps text-label-caps text-on-surface-variant">
                  MODEL_VERSION
                </span>
                <span className="material-symbols-outlined text-secondary-fixed-dim">memory</span>
              </div>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-surface-variant/10 border border-outline-variant/50 p-3 font-body-md text-on-surface focus:border-surface-tint focus:ring-0"
              >
                {modelVersions.map((model) => (
                  <option key={model.id} value={model.version}>
                    {model.version} - {model.accuracy}% acc
                  </option>
                ))}
              </select>
            </article>

            {/* Training Actions */}
            <article className="p-4 flex flex-col gap-3"
                     style={{
                       background: 'rgba(13, 18, 18, 0.7)',
                       backdropFilter: 'blur(12px)',
                       border: '0.5px solid rgba(185, 202, 203, 0.15)'
                     }}>
              <button
                onClick={handleTrainingToggle}
                className={`w-full p-3 border font-label-caps text-label-caps transition-all ${
                  isTraining
                    ? 'border-error/50 text-error hover:bg-error/10'
                    : 'border-secondary-fixed-dim/50 text-secondary-fixed-dim hover:bg-secondary-fixed-dim/10'
                }`}
              >
                {isTraining ? '[PAUSE_TRAINING]' : '[START_TRAINING]'}
              </button>
              <button className="w-full p-3 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors">
                [SAVE_CHECKPOINT]
              </button>
            </article>
          </div>

          {/* Fine-tuning Parameter Controls */}
          <section className="col-span-12 md:col-span-7"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center gap-3">
              <div className="w-2 h-6 bg-surface-tint"></div>
              <h3 className="font-headline-sm text-headline-sm">HYPERPARAMETER_TUNING</h3>
            </div>
            <div className="p-6 space-y-6">
              {/* Learning Rate */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-label-caps font-label-caps text-on-surface-variant">
                    LEARNING_RATE
                  </label>
                  <span className="font-data-display text-surface-tint">{learningRate.toFixed(4)}</span>
                </div>
                <input
                  type="range"
                  min="0.0001"
                  max="0.01"
                  step="0.0001"
                  value={learningRate}
                  onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                  className="w-full h-2 bg-outline-variant/30 appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-surface-tint [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-surface-tint"
                />
              </div>

              {/* Temperature */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-label-caps font-label-caps text-on-surface-variant">
                    TEMPERATURE
                  </label>
                  <span className="font-data-display text-surface-tint">{temperature.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="2.0"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full h-2 bg-outline-variant/30 appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-surface-tint [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-surface-tint"
                />
              </div>

              {/* Top P */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-label-caps font-label-caps text-on-surface-variant">
                    TOP_P (NUCLEUS_SAMPLING)
                  </label>
                  <span className="font-data-display text-surface-tint">{topP.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full h-2 bg-outline-variant/30 appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-surface-tint [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-surface-tint"
                />
              </div>

              {/* Max Tokens */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-label-caps font-label-caps text-on-surface-variant">
                    MAX_TOKENS
                  </label>
                  <span className="font-data-display text-surface-tint">{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="512"
                  max="8192"
                  step="256"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full h-2 bg-outline-variant/30 appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-surface-tint [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-surface-tint"
                />
              </div>
            </div>
          </section>

          {/* Training History Log */}
          <section className="col-span-12 md:col-span-5 flex flex-col h-[400px]"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex justify-between items-center">
              <span className="text-label-caps font-label-caps text-on-surface-variant">
                TRAINING_LOG::STREAM
              </span>
              <span className="text-label-caps font-label-caps text-surface-tint">LIVE</span>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3 font-body-md text-[12px] font-mono">
              {trainingLogs.map((log, index) => (
                <div key={index} className="flex gap-3 items-start">
                  <span className="text-on-surface-variant/40">[EP{log.epoch}]</span>
                  <span className={getLogColor(log.type)}>{log.type.toUpperCase()}</span>
                  <span className="text-on-surface-variant flex-1">{log.message}</span>
                  <span className="text-on-surface-variant/40 text-[10px]">
                    {formatTimestamp(log.timestamp)}
                  </span>
                </div>
              ))}
            </div>
            <div className="p-2 border-t border-outline-variant/30 bg-surface-container-low/50">
              <div className="flex justify-center items-center gap-4 py-2">
                <div className="w-1 h-1 bg-secondary-fixed-dim rounded-full animate-ping"></div>
                <span className="text-label-caps font-label-caps text-secondary-fixed-dim tracking-[0.3em]">
                  TRAINING_IN_PROGRESS
                </span>
              </div>
            </div>
          </section>

          {/* Training Dataset Management */}
          <section className="col-span-12 mt-4">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-surface-tint">database</span>
                <h3 className="font-label-caps text-label-caps text-surface-tint">
                  TRAINING_DATASETS // ACTIVE_CORPUS
                </h3>
              </div>
              <button className="px-4 py-2 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors">
                [ADD_DATASET]
              </button>
            </div>
            <div className="overflow-hidden"
                 style={{
                   background: 'rgba(13, 18, 18, 0.7)',
                   backdropFilter: 'blur(12px)',
                   border: '0.5px solid rgba(185, 202, 203, 0.15)'
                 }}>
              <table className="w-full">
                <thead className="border-b border-outline-variant/30">
                  <tr className="text-left">
                    <th className="p-4 font-label-caps text-label-caps text-on-surface-variant">DATASET_ID</th>
                    <th className="p-4 font-label-caps text-label-caps text-on-surface-variant">NAME</th>
                    <th className="p-4 font-label-caps text-label-caps text-on-surface-variant">SIZE</th>
                    <th className="p-4 font-label-caps text-label-caps text-on-surface-variant">SAMPLES</th>
                    <th className="p-4 font-label-caps text-label-caps text-on-surface-variant">STATUS</th>
                    <th className="p-4 font-label-caps text-label-caps text-on-surface-variant">ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((dataset) => (
                    <tr key={dataset.id} className="border-b border-outline-variant/10 hover:bg-surface-variant/10 transition-colors">
                      <td className="p-4 font-mono text-surface-tint">{dataset.id}</td>
                      <td className="p-4 text-on-surface">{dataset.name}</td>
                      <td className="p-4 text-on-surface-variant">{dataset.size}</td>
                      <td className="p-4 font-data-display text-on-surface">{dataset.samples.toLocaleString()}</td>
                      <td className="p-4">
                        <span className={`font-label-caps text-label-caps ${getStatusColor(dataset.status)}`}>
                          {dataset.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="p-4">
                        <div className="flex gap-2">
                          <button className="material-symbols-outlined text-surface-tint hover:text-secondary-fixed-dim transition-colors">
                            edit
                          </button>
                          <button className="material-symbols-outlined text-on-surface-variant hover:text-error transition-colors">
                            delete
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export { EchoTrainingInterface };
