import { useState, useEffect } from 'react';
import { useLabsHealth } from '../../lib/hooks';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

interface WorkloadMetric {
  label: string;
  value: string | number;
  unit?: string;
  status: 'normal' | 'warning' | 'critical';
  icon: string;
}

interface Task {
  id: string;
  name: string;
  status: 'active' | 'queued' | 'completed';
  progress: number;
  startTime?: string;
  completedTime?: string;
  priority: 'high' | 'medium' | 'low';
}

const EchoWorkloadDashboard = () => {
  const { labsStatus, loading } = useLabsHealth();
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Mock data for demonstration
  const [activeTasks] = useState<Task[]>([
    {
      id: 'task-001',
      name: 'Batch #409-Z Synthesis',
      status: 'active',
      progress: 62,
      startTime: '14:20:15',
      priority: 'high'
    },
    {
      id: 'task-002',
      name: 'Neural Cluster Alpha-9 Training',
      status: 'active',
      progress: 82,
      startTime: '13:45:22',
      priority: 'high'
    },
    {
      id: 'task-003',
      name: 'DNA Sequencer Calibration',
      status: 'active',
      progress: 45,
      startTime: '14:05:33',
      priority: 'medium'
    }
  ]);

  const [queuedTasks] = useState<Task[]>([
    {
      id: 'task-004',
      name: 'Protein Folding Analysis',
      status: 'queued',
      progress: 0,
      priority: 'medium'
    },
    {
      id: 'task-005',
      name: 'CRISPR Edit Verification',
      status: 'queued',
      progress: 0,
      priority: 'high'
    },
    {
      id: 'task-006',
      name: 'Metabolic Pathway Optimization',
      status: 'queued',
      progress: 0,
      priority: 'low'
    },
    {
      id: 'task-007',
      name: 'Drug Interaction Simulation',
      status: 'queued',
      progress: 0,
      priority: 'medium'
    }
  ]);

  const [completedTasks] = useState<Task[]>([
    {
      id: 'task-008',
      name: 'Node NA-01 Stabilization',
      status: 'completed',
      progress: 100,
      completedTime: '14:22:01',
      priority: 'high'
    },
    {
      id: 'task-009',
      name: 'EU Array Traffic Reroute',
      status: 'completed',
      progress: 100,
      completedTime: '14:22:18',
      priority: 'medium'
    },
    {
      id: 'task-010',
      name: 'APAC Cluster Sync',
      status: 'completed',
      progress: 100,
      completedTime: '13:58:42',
      priority: 'medium'
    }
  ]);

  const performanceMetrics: WorkloadMetric[] = [
    {
      label: 'FLEET_UTILIZATION',
      value: 84.2,
      unit: '%',
      status: 'normal',
      icon: 'rocket_launch'
    },
    {
      label: 'PROCESSING_SPEED',
      value: '4,029',
      unit: '/SEC',
      status: 'normal',
      icon: 'speed'
    },
    {
      label: 'AVG_LATENCY',
      value: 14,
      unit: 'MS',
      status: 'normal',
      icon: 'network_ping'
    },
    {
      label: 'ERROR_RATE',
      value: 0.02,
      unit: '%',
      status: 'normal',
      icon: 'bug_report'
    }
  ];

  const resourceMetrics: WorkloadMetric[] = [
    {
      label: 'CPU_USAGE',
      value: 67,
      unit: '%',
      status: 'normal',
      icon: 'memory'
    },
    {
      label: 'MEMORY',
      value: 72,
      unit: '%',
      status: 'warning',
      icon: 'storage'
    },
    {
      label: 'NETWORK_I/O',
      value: 89,
      unit: '%',
      status: 'warning',
      icon: 'swap_vert'
    },
    {
      label: 'DISK_USAGE',
      value: 54,
      unit: '%',
      status: 'normal',
      icon: 'hard_drive'
    }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return 'text-secondary-fixed-dim';
      case 'warning': return 'text-primary-fixed-dim';
      case 'critical': return 'text-error';
      default: return 'text-on-surface-variant';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-error border-error';
      case 'medium': return 'text-primary-fixed-dim border-primary-fixed-dim';
      case 'low': return 'text-on-surface-variant border-on-surface-variant';
      default: return 'text-on-surface-variant border-on-surface-variant';
    }
  };

  const getActiveCount = () => activeTasks.length;
  const getQueuedCount = () => queuedTasks.length;
  const getCompletedCount = () => completedTasks.length;
  const getOnlineCount = () => {
    if (loading) return 0;
    return Object.values(labsStatus).filter(lab => lab.healthy).length;
  };

  return (
    <div className="min-h-screen bg-background text-on-surface font-body-md">
      {/* Scanline Overlay */}
      <div className="fixed inset-0 pointer-events-none opacity-20 z-50"
           style={{
             background: 'linear-gradient(to bottom, transparent 50%, rgba(0, 219, 233, 0.03) 50%)',
             backgroundSize: '100% 4px'
           }}></div>

      {/* Top App Bar */}
      <header className="flex justify-between items-center w-full px-margin-mobile md:px-margin-desktop py-unit border-b border-outline-variant/50 bg-background/80 backdrop-blur-xl fixed top-0 z-40">
        <div className="flex items-center gap-3">
          <span className="material-symbols-outlined text-surface-tint">analytics</span>
          <h1 className="font-headline-sm text-headline-sm font-bold text-surface-tint tracking-tighter">
            ECHO_WORKLOAD_MONITOR // V.1.0.4
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-4 text-label-caps font-label-caps">
            <span className="text-on-surface-variant">{currentTime.toLocaleTimeString()}</span>
            <span className="text-on-surface-variant/40">|</span>
            <span className="text-secondary-fixed-dim">MONITORING_ACTIVE</span>
          </div>
        </div>
      </header>

            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-20 pb-8 px-margin-mobile md:px-margin-desktop min-h-screen">
        <div className="max-w-[1600px] mx-auto space-y-gutter">

          {/* Status Overview */}
          <section className="grid grid-cols-1 md:grid-cols-4 gap-tile-gap">
            <div className="p-4 border-l-4 border-surface-tint"
                 style={{
                   background: 'rgba(13, 18, 18, 0.7)',
                   backdropFilter: 'blur(12px)',
                   border: '0.5px solid rgba(185, 202, 203, 0.15)',
                   borderLeft: '4px solid #00dbe9'
                 }}>
              <span className="text-label-caps font-label-caps text-on-surface-variant block mb-2">
                ACTIVE_TASKS
              </span>
              <div className="font-data-display text-data-display text-surface-tint">
                {getActiveCount()}
              </div>
              <div className="text-[10px] text-on-surface-variant mt-1">CURRENTLY_PROCESSING</div>
            </div>

            <div className="p-4 border-l-4 border-primary-fixed-dim"
                 style={{
                   background: 'rgba(13, 18, 18, 0.7)',
                   backdropFilter: 'blur(12px)',
                   border: '0.5px solid rgba(185, 202, 203, 0.15)',
                   borderLeft: '4px solid #00dbe9'
                 }}>
              <span className="text-label-caps font-label-caps text-on-surface-variant block mb-2">
                QUEUED_TASKS
              </span>
              <div className="font-data-display text-data-display text-primary-fixed-dim">
                {getQueuedCount()}
              </div>
              <div className="text-[10px] text-on-surface-variant mt-1">WAITING_FOR_RESOURCES</div>
            </div>

            <div className="p-4 border-l-4 border-secondary-fixed-dim"
                 style={{
                   background: 'rgba(13, 18, 18, 0.7)',
                   backdropFilter: 'blur(12px)',
                   border: '0.5px solid rgba(185, 202, 203, 0.15)',
                   borderLeft: '4px solid #00e639'
                 }}>
              <span className="text-label-caps font-label-caps text-on-surface-variant block mb-2">
                COMPLETED_24H
              </span>
              <div className="font-data-display text-data-display text-secondary-fixed-dim">
                {getCompletedCount()}
              </div>
              <div className="text-[10px] text-on-surface-variant mt-1">LAST_24_HOURS</div>
            </div>

            <div className="p-4 border-l-4 border-secondary-fixed-dim"
                 style={{
                   background: 'rgba(13, 18, 18, 0.7)',
                   backdropFilter: 'blur(12px)',
                   border: '0.5px solid rgba(185, 202, 203, 0.15)',
                   borderLeft: '4px solid #00e639'
                 }}>
              <span className="text-label-caps font-label-caps text-on-surface-variant block mb-2">
                ONLINE_LABS
              </span>
              <div className="font-data-display text-data-display text-secondary-fixed-dim">
                {getOnlineCount()}
              </div>
              <div className="text-[10px] text-on-surface-variant mt-1">NODES_REACHABLE</div>
            </div>
          </section>

          {/* Active Tasks */}
          <section>
            <div className="flex items-center gap-2 mb-4">
              <span className="material-symbols-outlined text-surface-tint">play_circle</span>
              <h2 className="font-label-caps text-label-caps text-surface-tint">
                ACTIVE_TASKS // PROCESSING_NOW
              </h2>
            </div>
            <div className="space-y-tile-gap">
              {activeTasks.map((task) => (
                <div key={task.id}
                     className="p-4 border border-outline-variant/30 hover:border-surface-tint/50 transition-all"
                     style={{
                       background: 'rgba(13, 18, 18, 0.7)',
                       backdropFilter: 'blur(12px)'
                     }}>
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <h3 className="font-headline-sm text-headline-sm text-on-surface">
                          {task.name}
                        </h3>
                        <span className={`px-2 py-0.5 border font-label-caps text-[9px] ${getPriorityColor(task.priority)}`}>
                          {task.priority.toUpperCase()}
                        </span>
                      </div>
                      <div className="text-label-caps text-[10px] text-on-surface-variant">
                        STARTED: {task.startTime} | TASK_ID: {task.id.toUpperCase()}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-data-display text-headline-sm text-surface-tint">
                        {task.progress}%
                      </div>
                    </div>
                  </div>
                  <div className="w-full h-2 bg-outline-variant/30 relative overflow-hidden">
                    <div
                      className="h-full bg-surface-tint transition-all duration-500"
                      style={{ width: `${task.progress}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Processing Queue */}
          <section>
            <div className="flex items-center gap-2 mb-4">
              <span className="material-symbols-outlined text-primary-fixed-dim">pending</span>
              <h2 className="font-label-caps text-label-caps text-primary-fixed-dim">
                PROCESSING_QUEUE // PENDING_EXECUTION
              </h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-tile-gap">
              {queuedTasks.map((task, index) => (
                <div key={task.id}
                     className="p-4 border border-outline-variant/30"
                     style={{
                       background: 'rgba(13, 18, 18, 0.7)',
                       backdropFilter: 'blur(12px)'
                     }}>
                  <div className="flex justify-between items-start">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-label-caps text-[10px] text-on-surface-variant">
                          QUEUE_POS: #{index + 1}
                        </span>
                        <span className={`px-2 py-0.5 border font-label-caps text-[9px] ${getPriorityColor(task.priority)}`}>
                          {task.priority.toUpperCase()}
                        </span>
                      </div>
                      <h3 className="font-body-md text-on-surface">{task.name}</h3>
                      <div className="text-label-caps text-[10px] text-on-surface-variant mt-1">
                        {task.id.toUpperCase()}
                      </div>
                    </div>
                    <span className="material-symbols-outlined text-on-surface-variant/40">
                      schedule
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Performance & Resource Metrics */}
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-gutter">
            {/* Performance Metrics */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <span className="material-symbols-outlined text-secondary-fixed-dim">speed</span>
                <h2 className="font-label-caps text-label-caps text-secondary-fixed-dim">
                  PERFORMANCE_METRICS
                </h2>
              </div>
              <div className="grid grid-cols-2 gap-tile-gap">
                {performanceMetrics.map((metric) => (
                  <div key={metric.label}
                       className="p-4 border border-outline-variant/30"
                       style={{
                         background: 'rgba(13, 18, 18, 0.7)',
                         backdropFilter: 'blur(12px)'
                       }}>
                    <div className="flex justify-between items-start mb-2">
                      <span className="text-label-caps font-label-caps text-on-surface-variant">
                        {metric.label}
                      </span>
                      <span className={`material-symbols-outlined ${getStatusColor(metric.status)}`}>
                        {metric.icon}
                      </span>
                    </div>
                    <div className={`font-data-display text-data-display ${getStatusColor(metric.status)}`}>
                      {metric.value}{metric.unit && <span className="text-sm ml-1">{metric.unit}</span>}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Resource Utilization */}
            <div>
              <div className="flex items-center gap-2 mb-4">
                <span className="material-symbols-outlined text-primary-fixed-dim">memory</span>
                <h2 className="font-label-caps text-label-caps text-primary-fixed-dim">
                  RESOURCE_UTILIZATION
                </h2>
              </div>
              <div className="grid grid-cols-2 gap-tile-gap">
                {resourceMetrics.map((metric) => (
                  <div key={metric.label}
                       className="p-4 border border-outline-variant/30"
                       style={{
                         background: 'rgba(13, 18, 18, 0.7)',
                         backdropFilter: 'blur(12px)'
                       }}>
                    <div className="flex justify-between items-start mb-2">
                      <span className="text-label-caps font-label-caps text-on-surface-variant">
                        {metric.label}
                      </span>
                      <span className={`material-symbols-outlined ${getStatusColor(metric.status)}`}>
                        {metric.icon}
                      </span>
                    </div>
                    <div className={`font-data-display text-data-display ${getStatusColor(metric.status)}`}>
                      {metric.value}{metric.unit && <span className="text-sm ml-1">{metric.unit}</span>}
                    </div>
                    <div className="w-full h-1 bg-outline-variant/30 mt-2">
                      <div
                        className={`h-full transition-all ${
                          metric.status === 'warning' ? 'bg-primary-fixed-dim' :
                          metric.status === 'critical' ? 'bg-error' : 'bg-secondary-fixed-dim'
                        }`}
                        style={{ width: `${metric.value}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Completed Jobs (Last 24h) */}
          <section>
            <div className="flex items-center gap-2 mb-4">
              <span className="material-symbols-outlined text-secondary-fixed-dim">check_circle</span>
              <h2 className="font-label-caps text-label-caps text-secondary-fixed-dim">
                COMPLETED_JOBS // LAST_24_HOURS
              </h2>
            </div>
            <div className="p-4 border border-outline-variant/30"
                 style={{
                   background: 'rgba(13, 18, 18, 0.7)',
                   backdropFilter: 'blur(12px)'
                 }}>
              <div className="space-y-2">
                {completedTasks.map((task) => (
                  <div key={task.id}
                       className="flex justify-between items-center p-3 border-l-2 border-secondary-fixed-dim/50 bg-surface-container-lowest/30">
                    <div className="flex items-center gap-3">
                      <span className="material-symbols-outlined text-secondary-fixed-dim">done</span>
                      <div>
                        <div className="font-body-md text-on-surface">{task.name}</div>
                        <div className="text-label-caps text-[10px] text-on-surface-variant">
                          {task.id.toUpperCase()}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-label-caps text-[10px] text-on-surface-variant">
                        COMPLETED
                      </div>
                      <div className="font-body-md text-secondary-fixed-dim">
                        {task.completedTime}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export { EchoWorkloadDashboard };
