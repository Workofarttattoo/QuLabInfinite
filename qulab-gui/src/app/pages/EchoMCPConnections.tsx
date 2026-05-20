import { useState } from 'react';
import { useNavigate } from 'react-router';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

interface ConnectedApp {
  id: string;
  name: string;
  icon: string;
  status: 'connected' | 'syncing' | 'error' | 'disconnected';
  lastSync: string;
  dataFlow: string;
}

interface Webhook {
  id: string;
  url: string;
  eventType: string;
  status: 'active' | 'inactive';
  lastTriggered: string;
}

interface IntegrationLog {
  timestamp: string;
  app: string;
  event: string;
  status: 'success' | 'error' | 'pending';
}

const EchoMCPConnections = () => {
  const navigate = useNavigate();
  const [showAddModal, setShowAddModal] = useState<boolean>(false);
  const [newConnectionName, setNewConnectionName] = useState<string>('');
  const [newConnectionUrl, setNewConnectionUrl] = useState<string>('');
  const [selectedApp, setSelectedApp] = useState<string | null>(null);

  const [connectedApps] = useState<ConnectedApp[]>([
    {
      id: 'app-001',
      name: 'GitHub Enterprise',
      icon: 'code_blocks',
      status: 'connected',
      lastSync: '2m ago',
      dataFlow: 'bidirectional'
    },
    {
      id: 'app-002',
      name: 'Figma Design System',
      icon: 'palette',
      status: 'connected',
      lastSync: '5m ago',
      dataFlow: 'pull'
    },
    {
      id: 'app-003',
      name: 'Supabase Backend',
      icon: 'database',
      status: 'syncing',
      lastSync: '1m ago',
      dataFlow: 'push'
    },
    {
      id: 'app-004',
      name: 'Slack Workspace',
      icon: 'forum',
      status: 'error',
      lastSync: '2h ago',
      dataFlow: 'push'
    },
    {
      id: 'app-005',
      name: 'Linear Project Tracker',
      icon: 'checklist',
      status: 'disconnected',
      lastSync: '1d ago',
      dataFlow: 'bidirectional'
    },
    {
      id: 'app-006',
      name: 'Anthropic Claude API',
      icon: 'psychology',
      status: 'connected',
      lastSync: 'just now',
      dataFlow: 'bidirectional'
    },
  ]);

  const [webhooks, setWebhooks] = useState<Webhook[]>([
    {
      id: 'wh-001',
      url: 'https://echo.quantum.ai/webhooks/github-push',
      eventType: 'repository.push',
      status: 'active',
      lastTriggered: '2m ago'
    },
    {
      id: 'wh-002',
      url: 'https://echo.quantum.ai/webhooks/figma-update',
      eventType: 'design.updated',
      status: 'active',
      lastTriggered: '15m ago'
    },
    {
      id: 'wh-003',
      url: 'https://echo.quantum.ai/webhooks/slack-notify',
      eventType: 'message.posted',
      status: 'inactive',
      lastTriggered: '3h ago'
    },
  ]);

  const [integrationLogs] = useState<IntegrationLog[]>([
    { timestamp: '2026-05-18T14:32:15Z', app: 'GitHub', event: 'Repository commit pushed', status: 'success' },
    { timestamp: '2026-05-18T14:31:42Z', app: 'Figma', event: 'Component library synced', status: 'success' },
    { timestamp: '2026-05-18T14:30:28Z', app: 'Supabase', event: 'Database schema updated', status: 'pending' },
    { timestamp: '2026-05-18T14:29:53Z', app: 'Slack', event: 'OAuth token refresh failed', status: 'error' },
    { timestamp: '2026-05-18T14:28:17Z', app: 'Claude API', event: 'Model inference completed', status: 'success' },
    { timestamp: '2026-05-18T14:27:04Z', app: 'GitHub', event: 'Pull request created', status: 'success' },
    { timestamp: '2026-05-18T14:25:39Z', app: 'Linear', event: 'Issue status updated', status: 'success' },
    { timestamp: '2026-05-18T14:24:12Z', app: 'Figma', event: 'Design file accessed', status: 'success' },
  ]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'text-secondary-fixed-dim';
      case 'syncing': return 'text-surface-tint';
      case 'error': return 'text-error';
      case 'disconnected': return 'text-on-surface-variant';
      default: return 'text-on-surface-variant';
    }
  };

  const getStatusDot = (status: string) => {
    switch (status) {
      case 'connected': return 'bg-secondary-fixed-dim';
      case 'syncing': return 'bg-surface-tint';
      case 'error': return 'bg-error';
      case 'disconnected': return 'bg-on-surface-variant';
      default: return 'bg-on-surface-variant';
    }
  };

  const getLogStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'text-secondary-fixed-dim';
      case 'error': return 'text-error';
      case 'pending': return 'text-surface-tint';
      default: return 'text-on-surface-variant';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString();
  };

  const handleAddConnection = (e: React.FormEvent) => {
    e.preventDefault();
    console.log('Adding connection:', newConnectionName, newConnectionUrl);
    setShowAddModal(false);
    setNewConnectionName('');
    setNewConnectionUrl('');
  };

  const handleTestConnection = (appId: string) => {
    console.log('Testing connection:', appId);
  };

  const toggleWebhook = (id: string) => {
    setWebhooks(prev =>
      prev.map(wh => (wh.id === id ? { ...wh, status: wh.status === 'active' ? 'inactive' : 'active' } : wh))
    );
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
          <span className="material-symbols-outlined text-surface-tint">hub</span>
          <h1 className="font-headline-sm text-headline-sm font-bold text-surface-tint tracking-tighter">
            ECHO_MCP_CONNECTIONS // INTEGRATIONS
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-4 text-label-caps font-label-caps">
            <span className="text-secondary-fixed-dim">
              {connectedApps.filter(app => app.status === 'connected').length} ACTIVE
            </span>
            <span className="text-on-surface-variant/40">|</span>
            <span className="text-on-surface-variant">UPTIME: 99.94%</span>
          </div>
          <button className="px-3 py-1 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors">
            [MCP_PROTOCOL]
          </button>
        </div>
      </header>

      {/* Main Content Canvas */}
            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-20 pb-20 md:pb-8 min-h-screen">
        <div className="p-gutter grid grid-cols-12 gap-tile-gap">

          {/* Connected Apps Grid */}
          <section className="col-span-12">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-surface-tint">apps</span>
                <h3 className="font-label-caps text-label-caps text-surface-tint">
                  CONNECTED_APPLICATIONS // {connectedApps.length} TOTAL
                </h3>
              </div>
              <button
                onClick={() => setShowAddModal(true)}
                className="px-4 py-2 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors flex items-center gap-2"
              >
                <span className="material-symbols-outlined">add</span>
                [ADD_CONNECTION]
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {connectedApps.map((app) => (
                <div
                  key={app.id}
                  onClick={() => setSelectedApp(app.id)}
                  className={`p-4 cursor-pointer transition-all ${
                    selectedApp === app.id
                      ? 'border-surface-tint/60 bg-surface-tint/5'
                      : 'border-outline-variant/30 hover:border-outline-variant/50'
                  }`}
                  style={{
                    background: selectedApp === app.id ? 'rgba(0, 219, 233, 0.05)' : 'rgba(13, 18, 18, 0.7)',
                    backdropFilter: 'blur(12px)',
                    border: `0.5px solid ${selectedApp === app.id ? 'rgba(0, 219, 233, 0.6)' : 'rgba(185, 202, 203, 0.15)'}`
                  }}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <span className="material-symbols-outlined text-surface-tint text-[32px]">{app.icon}</span>
                      <div>
                        <div className="font-label-caps text-label-caps text-on-surface">{app.name}</div>
                        <div className="flex items-center gap-2 mt-1">
                          <div className={`w-1.5 h-1.5 rounded-full ${getStatusDot(app.status)} ${app.status === 'syncing' ? 'animate-pulse' : ''}`}></div>
                          <span className={`text-[10px] font-label-caps ${getStatusColor(app.status)}`}>
                            {app.status.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="text-[12px] text-on-surface-variant mb-3">
                    <div className="flex justify-between">
                      <span>Last sync:</span>
                      <span className="text-surface-tint">{app.lastSync}</span>
                    </div>
                    <div className="flex justify-between mt-1">
                      <span>Data flow:</span>
                      <span className="text-surface-tint uppercase">{app.dataFlow}</span>
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleTestConnection(app.id);
                    }}
                    className="w-full px-3 py-2 border border-outline-variant/50 text-on-surface-variant font-label-caps text-label-caps hover:bg-outline-variant/10 transition-colors"
                  >
                    [TEST_CONNECTION]
                  </button>
                </div>
              ))}
            </div>
          </section>

          {/* Data Flow Visualization */}
          <section className="col-span-12 lg:col-span-8"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center gap-3">
              <span className="material-symbols-outlined text-surface-tint">account_tree</span>
              <h3 className="font-headline-sm text-headline-sm">DATA_FLOW_VISUALIZATION</h3>
            </div>
            <div className="p-6 relative h-[400px] flex items-center justify-center">
              {/* Mock Data Flow Diagram */}
              <div className="relative w-full h-full">
                {/* Center Echo Node */}
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                  <div className="w-24 h-24 rounded-full border-2 border-surface-tint bg-surface-tint/10 flex items-center justify-center"
                       style={{ boxShadow: '0 0 20px rgba(0, 219, 233, 0.4)' }}>
                    <span className="material-symbols-outlined text-surface-tint text-[48px]">cloud_sync</span>
                  </div>
                  <div className="text-center mt-2">
                    <span className="font-label-caps text-label-caps text-surface-tint">ECHO_CORE</span>
                  </div>
                </div>

                {/* Source Nodes (Left) */}
                <div className="absolute top-[20%] left-[10%]">
                  <div className="w-16 h-16 rounded-full border border-secondary-fixed-dim/50 bg-secondary-fixed-dim/10 flex items-center justify-center">
                    <span className="material-symbols-outlined text-secondary-fixed-dim">code_blocks</span>
                  </div>
                  <div className="text-center mt-1">
                    <span className="text-[10px] text-on-surface-variant">GitHub</span>
                  </div>
                </div>
                <div className="absolute top-[60%] left-[10%]">
                  <div className="w-16 h-16 rounded-full border border-secondary-fixed-dim/50 bg-secondary-fixed-dim/10 flex items-center justify-center">
                    <span className="material-symbols-outlined text-secondary-fixed-dim">palette</span>
                  </div>
                  <div className="text-center mt-1">
                    <span className="text-[10px] text-on-surface-variant">Figma</span>
                  </div>
                </div>

                {/* Destination Nodes (Right) */}
                <div className="absolute top-[20%] right-[10%]">
                  <div className="w-16 h-16 rounded-full border border-surface-tint/50 bg-surface-tint/10 flex items-center justify-center">
                    <span className="material-symbols-outlined text-surface-tint">database</span>
                  </div>
                  <div className="text-center mt-1">
                    <span className="text-[10px] text-on-surface-variant">Supabase</span>
                  </div>
                </div>
                <div className="absolute top-[60%] right-[10%]">
                  <div className="w-16 h-16 rounded-full border border-surface-tint/50 bg-surface-tint/10 flex items-center justify-center">
                    <span className="material-symbols-outlined text-surface-tint">forum</span>
                  </div>
                  <div className="text-center mt-1">
                    <span className="text-[10px] text-on-surface-variant">Slack</span>
                  </div>
                </div>

                {/* Connection Lines (SVG) */}
                <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: -1 }}>
                  <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto">
                      <polygon points="0 0, 10 3, 0 6" fill="#00dbe9" opacity="0.5" />
                    </marker>
                  </defs>
                  {/* Left to Center Lines */}
                  <line x1="20%" y1="25%" x2="45%" y2="48%" stroke="#00e639" strokeWidth="2" strokeDasharray="5,5" opacity="0.4" markerEnd="url(#arrowhead)" />
                  <line x1="20%" y1="65%" x2="45%" y2="52%" stroke="#00e639" strokeWidth="2" strokeDasharray="5,5" opacity="0.4" markerEnd="url(#arrowhead)" />
                  {/* Center to Right Lines */}
                  <line x1="55%" y1="48%" x2="80%" y2="25%" stroke="#00dbe9" strokeWidth="2" strokeDasharray="5,5" opacity="0.4" markerEnd="url(#arrowhead)" />
                  <line x1="55%" y1="52%" x2="80%" y2="65%" stroke="#00dbe9" strokeWidth="2" strokeDasharray="5,5" opacity="0.4" markerEnd="url(#arrowhead)" />
                </svg>
              </div>

              {/* Legend */}
              <div className="absolute bottom-4 left-4 flex gap-6">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-0.5 bg-secondary-fixed-dim opacity-40"></div>
                  <span className="text-[10px] text-on-surface-variant">INBOUND</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-0.5 bg-surface-tint opacity-40"></div>
                  <span className="text-[10px] text-on-surface-variant">OUTBOUND</span>
                </div>
              </div>
            </div>
          </section>

          {/* Connection Health Monitoring */}
          <section className="col-span-12 lg:col-span-4"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center gap-3">
              <span className="material-symbols-outlined text-surface-tint">monitor_heart</span>
              <h3 className="font-headline-sm text-headline-sm">HEALTH_MONITOR</h3>
            </div>
            <div className="p-6 space-y-4">
              {connectedApps.slice(0, 4).map((app) => (
                <div key={app.id} className="p-3 border border-outline-variant/30 bg-surface-container-lowest/30">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-label-caps text-label-caps text-on-surface-variant">{app.name}</span>
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${getStatusDot(app.status)}`}></div>
                      <span className={`text-[10px] font-label-caps ${getStatusColor(app.status)}`}>
                        {app.status === 'connected' ? 'HEALTHY' : app.status.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  <div className="flex justify-between text-[10px] text-on-surface-variant/60">
                    <span>Ping:</span>
                    <span className="text-secondary-fixed-dim">{app.status === 'connected' ? '12ms' : 'N/A'}</span>
                  </div>
                  <div className="flex justify-between text-[10px] text-on-surface-variant/60 mt-1">
                    <span>Uptime:</span>
                    <span className="text-secondary-fixed-dim">{app.status === 'connected' ? '99.8%' : '0%'}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Webhook Configuration */}
          <section className="col-span-12"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-surface-tint">webhook</span>
                <h3 className="font-headline-sm text-headline-sm">WEBHOOK_CONFIGURATION</h3>
              </div>
              <button className="px-4 py-2 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors">
                [CREATE_WEBHOOK]
              </button>
            </div>
            <div className="overflow-hidden">
              <table className="w-full">
                <thead className="border-b border-outline-variant/30">
                  <tr className="text-left">
                    <th className="p-4 font-label-caps text-label-caps text-on-surface-variant">URL</th>
                    <th className="p-4 font-label-caps text-label-caps text-on-surface-variant">EVENT_TYPE</th>
                    <th className="p-4 font-label-caps text-label-caps text-on-surface-variant">STATUS</th>
                    <th className="p-4 font-label-caps text-label-caps text-on-surface-variant">LAST_TRIGGERED</th>
                    <th className="p-4 font-label-caps text-label-caps text-on-surface-variant">ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  {webhooks.map((webhook) => (
                    <tr key={webhook.id} className="border-b border-outline-variant/10 hover:bg-surface-variant/10 transition-colors">
                      <td className="p-4 font-mono text-[12px] text-surface-tint">{webhook.url}</td>
                      <td className="p-4 text-on-surface-variant">{webhook.eventType}</td>
                      <td className="p-4">
                        <button
                          onClick={() => toggleWebhook(webhook.id)}
                          className={`font-label-caps text-label-caps ${
                            webhook.status === 'active' ? 'text-secondary-fixed-dim' : 'text-on-surface-variant'
                          }`}
                        >
                          {webhook.status.toUpperCase()}
                        </button>
                      </td>
                      <td className="p-4 text-on-surface-variant">{webhook.lastTriggered}</td>
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

          {/* Integration Logs */}
          <section className="col-span-12 flex flex-col h-[400px]"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-surface-tint">description</span>
                <h3 className="font-headline-sm text-headline-sm">INTEGRATION_LOGS</h3>
              </div>
              <span className="text-label-caps font-label-caps text-surface-tint">LIVE_STREAM</span>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-2">
              {integrationLogs.map((log, index) => (
                <div key={index} className="flex items-center gap-4 p-3 border border-outline-variant/20 bg-surface-container-lowest/30 hover:bg-surface-variant/10 transition-colors">
                  <span className="text-[10px] text-on-surface-variant/60 w-20">{formatTimestamp(log.timestamp)}</span>
                  <div className="w-20">
                    <span className="text-label-caps text-label-caps text-surface-tint">{log.app}</span>
                  </div>
                  <div className="flex-1">
                    <span className="text-[12px] text-on-surface-variant">{log.event}</span>
                  </div>
                  <div className="w-24 text-right">
                    <span className={`text-label-caps text-label-caps ${getLogStatusColor(log.status)}`}>
                      {log.status.toUpperCase()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
            <div className="p-2 border-t border-outline-variant/30 bg-surface-container-low/50">
              <div className="flex justify-center items-center gap-4 py-2">
                <div className="w-1 h-1 bg-secondary-fixed-dim rounded-full animate-ping"></div>
                <span className="text-label-caps font-label-caps text-secondary-fixed-dim tracking-[0.3em]">
                  MONITORING_ACTIVE
                </span>
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* Add Connection Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-xl z-50 flex items-center justify-center p-4">
          <div className="max-w-2xl w-full"
               style={{
                 background: 'rgba(13, 18, 18, 0.95)',
                 backdropFilter: 'blur(12px)',
                 border: '0.5px solid rgba(0, 219, 233, 0.6)',
                 boxShadow: '0 0 20px rgba(0, 219, 233, 0.2)'
               }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center justify-between">
              <h3 className="font-headline-sm text-headline-sm text-surface-tint">ADD_NEW_CONNECTION</h3>
              <button
                onClick={() => setShowAddModal(false)}
                className="material-symbols-outlined text-on-surface-variant hover:text-error transition-colors"
              >
                close
              </button>
            </div>
            <form onSubmit={handleAddConnection} className="p-6 space-y-6">
              <div>
                <label className="text-label-caps font-label-caps text-on-surface-variant block mb-2">
                  APPLICATION_NAME
                </label>
                <input
                  type="text"
                  value={newConnectionName}
                  onChange={(e) => setNewConnectionName(e.target.value)}
                  className="w-full bg-surface-variant/10 border border-outline-variant/50 p-3 font-body-md text-on-surface focus:border-surface-tint focus:ring-0"
                  placeholder="e.g., Custom MCP Server"
                  required
                />
              </div>
              <div>
                <label className="text-label-caps font-label-caps text-on-surface-variant block mb-2">
                  CONNECTION_URL
                </label>
                <input
                  type="url"
                  value={newConnectionUrl}
                  onChange={(e) => setNewConnectionUrl(e.target.value)}
                  className="w-full bg-surface-variant/10 border border-outline-variant/50 p-3 font-body-md text-on-surface focus:border-surface-tint focus:ring-0"
                  placeholder="https://api.example.com"
                  required
                />
              </div>
              <div>
                <label className="text-label-caps font-label-caps text-on-surface-variant block mb-2">
                  AUTHENTICATION_TYPE
                </label>
                <select className="w-full bg-surface-variant/10 border border-outline-variant/50 p-3 font-body-md text-on-surface focus:border-surface-tint focus:ring-0">
                  <option value="oauth">OAuth 2.0</option>
                  <option value="apikey">API Key</option>
                  <option value="bearer">Bearer Token</option>
                  <option value="basic">Basic Auth</option>
                </select>
              </div>
              <div className="flex gap-4 justify-end pt-4">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="px-6 py-3 border border-outline-variant/50 text-on-surface-variant font-label-caps text-label-caps hover:bg-outline-variant/10 transition-colors"
                >
                  [CANCEL]
                </button>
                <button
                  type="submit"
                  className="px-6 py-3 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors"
                >
                  [CREATE_CONNECTION]
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

export { EchoMCPConnections };
