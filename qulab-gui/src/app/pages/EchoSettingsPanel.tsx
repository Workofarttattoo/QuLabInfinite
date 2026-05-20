import { useState } from 'react';
import { useNavigate } from 'react-router';
import { EchoLabCommandInline } from '../components/EchoLabCommandInline';

interface ApiKey {
  id: string;
  name: string;
  key: string;
  created: string;
  lastUsed: string;
}

interface EnvVariable {
  id: string;
  key: string;
  value: string;
}

interface Integration {
  id: string;
  name: string;
  icon: string;
  enabled: boolean;
  status: 'online' | 'offline' | 'error';
}

interface ChangeLog {
  timestamp: string;
  user: string;
  action: string;
  target: string;
}

const EchoSettingsPanel = () => {
  const navigate = useNavigate();
  const [theme, setTheme] = useState<string>('dark');
  const [notifications, setNotifications] = useState<boolean>(true);
  const [autoSave, setAutoSave] = useState<boolean>(true);
  const [encryptionEnabled, setEncryptionEnabled] = useState<boolean>(true);
  const [rateLimit, setRateLimit] = useState<number>(1000);
  const [showApiKeys, setShowApiKeys] = useState<{ [key: string]: boolean }>({});
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  const [apiKeys] = useState<ApiKey[]>([
    {
      id: 'ak-001',
      name: 'QUANTUM_BRIDGE_API',
      key: 'qb_sk_live_8x9k2p4m6n1v3c7z',
      created: '2026-05-10T08:30:00Z',
      lastUsed: '2m ago'
    },
    {
      id: 'ak-002',
      name: 'NEURAL_SYNC_TOKEN',
      key: 'ns_prod_4j7h9k2m5p8r1t3w',
      created: '2026-05-05T14:15:00Z',
      lastUsed: '45m ago'
    },
    {
      id: 'ak-003',
      name: 'ECHO_CORE_SECRET',
      key: 'ec_dev_2x5c8v1b4n7m9k3p',
      created: '2026-04-28T11:00:00Z',
      lastUsed: '3h ago'
    },
  ]);

  const [envVariables, setEnvVariables] = useState<EnvVariable[]>([
    { id: 'ev-001', key: 'DATABASE_URL', value: 'postgresql://echo:***@quantum.db:5432/neural' },
    { id: 'ev-002', key: 'REDIS_HOST', value: 'redis://cache.echo.internal:6379' },
    { id: 'ev-003', key: 'NODE_ENV', value: 'production' },
    { id: 'ev-004', key: 'LOG_LEVEL', value: 'debug' },
  ]);

  const [integrations, setIntegrations] = useState<Integration[]>([
    { id: 'int-001', name: 'GitHub MCP Server', icon: 'code', enabled: true, status: 'online' },
    { id: 'int-002', name: 'Figma Design Sync', icon: 'palette', enabled: true, status: 'online' },
    { id: 'int-003', name: 'Slack Notifications', icon: 'chat', enabled: false, status: 'offline' },
    { id: 'int-004', name: 'Supabase Backend', icon: 'storage', enabled: true, status: 'error' },
    { id: 'int-005', name: 'Anthropic Claude API', icon: 'psychology', enabled: true, status: 'online' },
    { id: 'int-006', name: 'Linear Issue Tracking', icon: 'task', enabled: false, status: 'offline' },
  ]);

  const [changeLogs] = useState<ChangeLog[]>([
    { timestamp: '2026-05-18T14:25:00Z', user: 'admin', action: 'Updated', target: 'Rate limit: 1000 req/min' },
    { timestamp: '2026-05-18T13:42:00Z', user: 'admin', action: 'Enabled', target: 'Integration: GitHub MCP' },
    { timestamp: '2026-05-18T12:15:00Z', user: 'system', action: 'Rotated', target: 'API Key: QUANTUM_BRIDGE' },
    { timestamp: '2026-05-18T10:30:00Z', user: 'admin', action: 'Modified', target: 'ENV: DATABASE_URL' },
    { timestamp: '2026-05-17T22:18:00Z', user: 'admin', action: 'Disabled', target: 'Integration: Slack' },
  ]);

  const toggleApiKeyVisibility = (id: string) => {
    setShowApiKeys(prev => ({ ...prev, [id]: !prev[id] }));
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const toggleIntegration = (id: string) => {
    setIntegrations(prev =>
      prev.map(int => (int.id === id ? { ...int, enabled: !int.enabled } : int))
    );
  };

  const handleSaveChanges = () => {
    setSaveStatus('saving');
    setTimeout(() => {
      setSaveStatus('success');
      setTimeout(() => setSaveStatus('idle'), 2000);
    }, 1500);
  };

  const handleResetDefaults = () => {
    if (confirm('Reset all settings to default values?')) {
      setTheme('dark');
      setNotifications(true);
      setAutoSave(true);
      setRateLimit(1000);
    }
  };

  const addEnvVariable = () => {
    const newId = `ev-${String(envVariables.length + 1).padStart(3, '0')}`;
    setEnvVariables([...envVariables, { id: newId, key: 'NEW_VARIABLE', value: '' }]);
  };

  const updateEnvVariable = (id: string, field: 'key' | 'value', newValue: string) => {
    setEnvVariables(prev =>
      prev.map(env => (env.id === id ? { ...env, [field]: newValue } : env))
    );
  };

  const deleteEnvVariable = (id: string) => {
    setEnvVariables(prev => prev.filter(env => env.id !== id));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'text-secondary-fixed-dim';
      case 'offline': return 'text-on-surface-variant';
      case 'error': return 'text-error';
      default: return 'text-on-surface-variant';
    }
  };

  const getStatusDot = (status: string) => {
    switch (status) {
      case 'online': return 'bg-secondary-fixed-dim';
      case 'offline': return 'bg-on-surface-variant';
      case 'error': return 'bg-error';
      default: return 'bg-on-surface-variant';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000 / 60);
    if (diff < 1) return 'just now';
    if (diff < 60) return `${diff}m ago`;
    if (diff < 1440) return `${Math.floor(diff / 60)}h ago`;
    return date.toLocaleDateString();
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
          <span className="material-symbols-outlined text-surface-tint">settings</span>
          <h1 className="font-headline-sm text-headline-sm font-bold text-surface-tint tracking-tighter">
            ECHO_SETTINGS_PANEL // CONFIGURATION
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-4 text-label-caps font-label-caps">
            <span className={`${encryptionEnabled ? 'text-secondary-fixed-dim' : 'text-error'}`}>
              {encryptionEnabled ? 'ENCRYPTED' : 'UNSECURED'}
            </span>
            <span className="text-on-surface-variant/40">|</span>
            <span className="text-on-surface-variant">AUTO_SAVE: {autoSave ? 'ON' : 'OFF'}</span>
          </div>
          <button className="px-3 py-1 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors">
            [ADMIN_MODE]
          </button>
        </div>
      </header>

      {/* Main Content Canvas */}
            <EchoLabCommandInline className="mb-6 px-4 md:px-8 max-w-[1440px] mx-auto w-full" />

      <main className="pt-20 pb-20 md:pb-8 min-h-screen">
        <div className="p-gutter grid grid-cols-12 gap-tile-gap">

          {/* System Preferences */}
          <section className="col-span-12 lg:col-span-6"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center gap-3">
              <div className="w-2 h-6 bg-surface-tint"></div>
              <h3 className="font-headline-sm text-headline-sm">SYSTEM_PREFERENCES</h3>
            </div>
            <div className="p-6 space-y-6">
              {/* Theme Selector */}
              <div>
                <label className="text-label-caps font-label-caps text-on-surface-variant block mb-3">
                  INTERFACE_THEME
                </label>
                <select
                  value={theme}
                  onChange={(e) => setTheme(e.target.value)}
                  className="w-full bg-surface-variant/10 border border-outline-variant/50 p-3 font-body-md text-on-surface focus:border-surface-tint focus:ring-0"
                >
                  <option value="dark">Dark Mode (Neural Grid)</option>
                  <option value="light">Light Mode (Quantum Field)</option>
                  <option value="auto">Auto (System Sync)</option>
                </select>
              </div>

              {/* Toggle Switches */}
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 border border-outline-variant/30 bg-surface-container-lowest/30">
                  <div>
                    <div className="font-label-caps text-label-caps text-on-surface-variant">NOTIFICATIONS</div>
                    <div className="text-[12px] text-on-surface-variant/60 mt-1">
                      Enable real-time system alerts
                    </div>
                  </div>
                  <button
                    onClick={() => setNotifications(!notifications)}
                    className={`relative w-14 h-7 rounded-full transition-colors ${
                      notifications ? 'bg-secondary-fixed-dim' : 'bg-outline-variant/50'
                    }`}
                  >
                    <div
                      className={`absolute top-1 w-5 h-5 bg-background rounded-full transition-transform ${
                        notifications ? 'translate-x-8' : 'translate-x-1'
                      }`}
                    ></div>
                  </button>
                </div>

                <div className="flex items-center justify-between p-4 border border-outline-variant/30 bg-surface-container-lowest/30">
                  <div>
                    <div className="font-label-caps text-label-caps text-on-surface-variant">AUTO_SAVE</div>
                    <div className="text-[12px] text-on-surface-variant/60 mt-1">
                      Automatically save configuration changes
                    </div>
                  </div>
                  <button
                    onClick={() => setAutoSave(!autoSave)}
                    className={`relative w-14 h-7 rounded-full transition-colors ${
                      autoSave ? 'bg-secondary-fixed-dim' : 'bg-outline-variant/50'
                    }`}
                  >
                    <div
                      className={`absolute top-1 w-5 h-5 bg-background rounded-full transition-transform ${
                        autoSave ? 'translate-x-8' : 'translate-x-1'
                      }`}
                    ></div>
                  </button>
                </div>
              </div>
            </div>
          </section>

          {/* Security Settings */}
          <section className="col-span-12 lg:col-span-6"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center gap-3">
              <div className="w-2 h-6 bg-secondary-fixed-dim"></div>
              <h3 className="font-headline-sm text-headline-sm">SECURITY_CONTROLS</h3>
            </div>
            <div className="p-6 space-y-6">
              {/* Encryption Status */}
              <div className="p-4 border border-outline-variant/30 bg-surface-container-lowest/30">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-label-caps text-label-caps text-on-surface-variant">
                    DATA_ENCRYPTION
                  </span>
                  <div className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${encryptionEnabled ? 'bg-secondary-fixed-dim' : 'bg-error'}`}></div>
                    <span className={`text-label-caps font-label-caps ${encryptionEnabled ? 'text-secondary-fixed-dim' : 'text-error'}`}>
                      {encryptionEnabled ? 'ACTIVE' : 'DISABLED'}
                    </span>
                  </div>
                </div>
                <div className="text-[12px] text-on-surface-variant/60">
                  AES-256-GCM | RSA-4096 Key Exchange
                </div>
                <button
                  onClick={() => setEncryptionEnabled(!encryptionEnabled)}
                  className="mt-3 w-full px-4 py-2 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors"
                >
                  {encryptionEnabled ? '[DISABLE_ENCRYPTION]' : '[ENABLE_ENCRYPTION]'}
                </button>
              </div>

              {/* Rate Limiting */}
              <div>
                <div className="flex justify-between items-center mb-3">
                  <label className="text-label-caps font-label-caps text-on-surface-variant">
                    RATE_LIMIT (REQ/MIN)
                  </label>
                  <span className="font-data-display text-surface-tint">{rateLimit}</span>
                </div>
                <input
                  type="range"
                  min="100"
                  max="5000"
                  step="100"
                  value={rateLimit}
                  onChange={(e) => setRateLimit(parseInt(e.target.value))}
                  className="w-full h-2 bg-outline-variant/30 appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:bg-surface-tint [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-surface-tint"
                />
                <div className="flex justify-between text-[10px] text-on-surface-variant/60 mt-1">
                  <span>100</span>
                  <span>5000</span>
                </div>
              </div>
            </div>
          </section>

          {/* API Key Management */}
          <section className="col-span-12"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-surface-tint">vpn_key</span>
                <h3 className="font-headline-sm text-headline-sm">API_KEY_MANAGEMENT</h3>
              </div>
              <button className="px-4 py-2 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors">
                [GENERATE_NEW_KEY]
              </button>
            </div>
            <div className="p-6 space-y-4">
              {apiKeys.map((apiKey) => (
                <div key={apiKey.id} className="p-4 border border-outline-variant/30 bg-surface-container-lowest/30">
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-label-caps text-label-caps text-surface-tint">{apiKey.name}</span>
                    <div className="flex gap-2">
                      <button
                        onClick={() => toggleApiKeyVisibility(apiKey.id)}
                        className="material-symbols-outlined text-on-surface-variant hover:text-surface-tint transition-colors"
                      >
                        {showApiKeys[apiKey.id] ? 'visibility_off' : 'visibility'}
                      </button>
                      <button
                        onClick={() => copyToClipboard(apiKey.key)}
                        className="material-symbols-outlined text-on-surface-variant hover:text-surface-tint transition-colors"
                      >
                        content_copy
                      </button>
                      <button className="material-symbols-outlined text-on-surface-variant hover:text-error transition-colors">
                        delete
                      </button>
                    </div>
                  </div>
                  <div className="bg-surface-variant/10 border border-outline-variant/50 p-3 font-mono text-[12px] text-on-surface">
                    {showApiKeys[apiKey.id] ? apiKey.key : '•'.repeat(apiKey.key.length)}
                  </div>
                  <div className="flex justify-between mt-2 text-[10px] text-on-surface-variant/60">
                    <span>Created: {new Date(apiKey.created).toLocaleDateString()}</span>
                    <span>Last used: {apiKey.lastUsed}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Environment Variables */}
          <section className="col-span-12 lg:col-span-7"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="material-symbols-outlined text-surface-tint">code</span>
                <h3 className="font-headline-sm text-headline-sm">ENVIRONMENT_VARIABLES</h3>
              </div>
              <button
                onClick={addEnvVariable}
                className="px-4 py-2 border border-surface-tint/50 text-surface-tint font-label-caps text-label-caps hover:bg-surface-tint/10 transition-colors"
              >
                [ADD_VARIABLE]
              </button>
            </div>
            <div className="p-6 space-y-3 max-h-[500px] overflow-y-auto">
              {envVariables.map((env) => (
                <div key={env.id} className="grid grid-cols-12 gap-3 items-center p-3 border border-outline-variant/30 bg-surface-container-lowest/30">
                  <input
                    type="text"
                    value={env.key}
                    onChange={(e) => updateEnvVariable(env.id, 'key', e.target.value)}
                    className="col-span-4 bg-surface-variant/10 border border-outline-variant/50 p-2 font-mono text-[12px] text-on-surface focus:border-surface-tint focus:ring-0"
                    placeholder="KEY"
                  />
                  <input
                    type="text"
                    value={env.value}
                    onChange={(e) => updateEnvVariable(env.id, 'value', e.target.value)}
                    className="col-span-7 bg-surface-variant/10 border border-outline-variant/50 p-2 font-mono text-[12px] text-on-surface focus:border-surface-tint focus:ring-0"
                    placeholder="VALUE"
                  />
                  <button
                    onClick={() => deleteEnvVariable(env.id)}
                    className="col-span-1 material-symbols-outlined text-on-surface-variant hover:text-error transition-colors"
                  >
                    delete
                  </button>
                </div>
              ))}
            </div>
          </section>

          {/* Recent Changes Log */}
          <section className="col-span-12 lg:col-span-5 flex flex-col h-[500px]"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center gap-3">
              <span className="material-symbols-outlined text-surface-tint">history</span>
              <h3 className="font-headline-sm text-headline-sm">RECENT_CHANGES</h3>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {changeLogs.map((log, index) => (
                <div key={index} className="p-3 border border-outline-variant/20 bg-surface-container-lowest/30">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-label-caps font-label-caps text-surface-tint">{log.action}</span>
                    <span className="text-[10px] text-on-surface-variant/60">{formatTimestamp(log.timestamp)}</span>
                  </div>
                  <div className="text-[12px] text-on-surface-variant">{log.target}</div>
                  <div className="text-[10px] text-on-surface-variant/40 mt-1">by: {log.user}</div>
                </div>
              ))}
            </div>
          </section>

          {/* Integration Toggles */}
          <section className="col-span-12"
                   style={{
                     background: 'rgba(13, 18, 18, 0.7)',
                     backdropFilter: 'blur(12px)',
                     border: '0.5px solid rgba(185, 202, 203, 0.15)'
                   }}>
            <div className="p-4 border-b border-outline-variant/30 flex items-center gap-3">
              <span className="material-symbols-outlined text-surface-tint">extension</span>
              <h3 className="font-headline-sm text-headline-sm">MCP_SERVER_INTEGRATIONS</h3>
            </div>
            <div className="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {integrations.map((integration) => (
                <div key={integration.id} className="p-4 border border-outline-variant/30 bg-surface-container-lowest/30">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-3">
                      <span className="material-symbols-outlined text-surface-tint">{integration.icon}</span>
                      <div>
                        <div className="font-label-caps text-label-caps text-on-surface">{integration.name}</div>
                        <div className="flex items-center gap-2 mt-1">
                          <div className={`w-1.5 h-1.5 rounded-full ${getStatusDot(integration.status)}`}></div>
                          <span className={`text-[10px] font-label-caps ${getStatusColor(integration.status)}`}>
                            {integration.status.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={() => toggleIntegration(integration.id)}
                    className={`w-full px-3 py-2 border font-label-caps text-label-caps transition-colors ${
                      integration.enabled
                        ? 'border-secondary-fixed-dim/50 text-secondary-fixed-dim hover:bg-secondary-fixed-dim/10'
                        : 'border-outline-variant/50 text-on-surface-variant hover:bg-outline-variant/10'
                    }`}
                  >
                    {integration.enabled ? '[ENABLED]' : '[DISABLED]'}
                  </button>
                </div>
              ))}
            </div>
          </section>

          {/* Action Buttons */}
          <section className="col-span-12 flex gap-4 justify-end">
            <button
              onClick={handleResetDefaults}
              className="px-6 py-3 border border-outline-variant/50 text-on-surface-variant font-label-caps text-label-caps hover:bg-outline-variant/10 transition-colors"
            >
              [RESET_TO_DEFAULT]
            </button>
            <button
              onClick={handleSaveChanges}
              disabled={saveStatus === 'saving'}
              className={`px-6 py-3 border font-label-caps text-label-caps transition-all ${
                saveStatus === 'success'
                  ? 'border-secondary-fixed-dim/50 text-secondary-fixed-dim bg-secondary-fixed-dim/10'
                  : saveStatus === 'error'
                  ? 'border-error/50 text-error bg-error/10'
                  : 'border-surface-tint/50 text-surface-tint hover:bg-surface-tint/10'
              }`}
            >
              {saveStatus === 'saving'
                ? '[SAVING...]'
                : saveStatus === 'success'
                ? '[SAVED_SUCCESS]'
                : saveStatus === 'error'
                ? '[SAVE_FAILED]'
                : '[SAVE_CHANGES]'}
            </button>
          </section>
        </div>
      </main>
    </div>
  );
};

export { EchoSettingsPanel };
