// QuLab Infinite API Client
// MCP HTTP: Port 8102 (primary agent/tool gateway for Materials & R&D)
// Unified API: Port 8000 (REST routes for lab workflows)
// Medical Labs: Ports 8001-8010 (standalone microservices)

const MCP_API_URL = import.meta.env.VITE_MCP_API_URL || '/mcp';
const UNIFIED_API_URL = import.meta.env.VITE_UNIFIED_API_URL || 'http://localhost:8000';
const MEDICAL_BASE_URL = import.meta.env.VITE_MEDICAL_BASE_URL || 'http://localhost';
const API_KEY = import.meta.env.VITE_API_KEY || '';
const MCP_API_KEY = import.meta.env.VITE_MCP_API_KEY || '';

export interface LabConfig {
  name: string;
  type: 'medical' | 'unified';
  port?: number; // Only for medical labs
  endpoint: string;
  category: 'medical' | 'materials' | 'quantum' | 'bio' | 'drug';
}

// QuLab Infinite Production Labs Configuration
export const LABS: Record<string, LabConfig> = {
  // === CORE R&D LABS (Unified API - Port 8000) ===
  materials: {
    name: 'Materials Science Lab',
    type: 'unified',
    endpoint: '/materials/analyze',
    category: 'materials'
  },
  quantum: {
    name: 'Quantum Computing Lab',
    type: 'unified',
    endpoint: '/quantum/simulate',
    category: 'quantum'
  },
  chemistry: {
    name: 'Computational Chemistry Lab',
    type: 'unified',
    endpoint: '/chemistry/synthesize',
    category: 'materials'
  },
  oncology: {
    name: 'Oncology Simulation Lab',
    type: 'unified',
    endpoint: '/oncology/simulate',
    category: 'bio'
  },
  drug: {
    name: 'Drug Discovery Lab',
    type: 'unified',
    endpoint: '/drug/screen',
    category: 'drug'
  },
  genomics: {
    name: 'Genomics Analysis Lab',
    type: 'unified',
    endpoint: '/genomics/analyze',
    category: 'bio'
  },
  immune: {
    name: 'Immune Response Lab',
    type: 'unified',
    endpoint: '/immune/simulate',
    category: 'bio'
  },
  metabolic: {
    name: 'Metabolic Syndrome Lab',
    type: 'unified',
    endpoint: '/metabolic/analyze',
    category: 'bio'
  },

  // === MEDICAL DIAGNOSTIC LABS (Standalone - Ports 8001-8010) ===
  alzheimers: {
    name: "Alzheimer's Early Detection",
    type: 'medical',
    port: 8001,
    endpoint: '/assess',
    category: 'medical'
  },
  parkinsons: {
    name: "Parkinson's Progression Predictor",
    type: 'medical',
    port: 8002,
    endpoint: '/assess',
    category: 'medical'
  },
  autoimmune: {
    name: 'Autoimmune Disease Classifier',
    type: 'medical',
    port: 8003,
    endpoint: '/assess',
    category: 'medical'
  },
  sepsis: {
    name: 'Sepsis Early Warning System',
    type: 'medical',
    port: 8004,
    endpoint: '/assess',
    category: 'medical'
  },
  wound: {
    name: 'Wound Healing Optimizer',
    type: 'medical',
    port: 8005,
    endpoint: '/assess',
    category: 'medical'
  },
  bone: {
    name: 'Bone Density Predictor',
    type: 'medical',
    port: 8006,
    endpoint: '/assess',
    category: 'medical'
  },
  kidney: {
    name: 'Kidney Function Calculator',
    type: 'medical',
    port: 8007,
    endpoint: '/assess',
    category: 'medical'
  },
  liver: {
    name: 'Liver Disease Staging System',
    type: 'medical',
    port: 8008,
    endpoint: '/assess',
    category: 'medical'
  },
  lung: {
    name: 'Lung Function Analyzer',
    type: 'medical',
    port: 8009,
    endpoint: '/assess',
    category: 'medical'
  },
  pain: {
    name: 'Pain Management Optimizer',
    type: 'medical',
    port: 8010,
    endpoint: '/assess',
    category: 'medical'
  },
};

class APIClient {
  private mcpApiUrl: string;
  private unifiedApiUrl: string;
  private medicalBaseUrl: string;
  private apiKey: string;
  private mcpApiKey: string;

  constructor() {
    this.mcpApiUrl = MCP_API_URL;
    this.unifiedApiUrl = UNIFIED_API_URL;
    this.medicalBaseUrl = MEDICAL_BASE_URL;
    this.apiKey = API_KEY;
    this.mcpApiKey = MCP_API_KEY || API_KEY; // Fallback to main API key
  }

  private async request<T>(
    url: string,
    method: 'GET' | 'POST' = 'GET',
    body?: any,
    useXApiKey: boolean = false
  ): Promise<T> {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };

    if (this.apiKey) {
      // Unified API uses X-Api-Key, Medical labs use Bearer
      if (useXApiKey) {
        headers['X-Api-Key'] = this.apiKey;
      } else {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }
    }

    const options: RequestInit = {
      method,
      headers,
      ...(body && { body: JSON.stringify(body) }),
    };

    try {
      const response = await fetch(url, options);

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      // Suppress console errors for health checks - expected when servers are offline
      if (!url.includes('/health')) {
        console.error(`Failed to fetch from ${url}:`, error);
      }
      throw error;
    }
  }

  private getLabUrl(lab: LabConfig, endpoint: string): string {
    if (lab.type === 'unified') {
      return `${this.unifiedApiUrl}${endpoint}`;
    } else {
      return `${this.medicalBaseUrl}:${lab.port}${endpoint}`;
    }
  }

  // Health check for a specific lab
  async checkHealth(labKey: string): Promise<{ status: string; lab?: string }> {
    const lab = LABS[labKey];
    if (!lab) throw new Error(`Unknown lab: ${labKey}`);

    const url = this.getLabUrl(lab, '/health');
    const useXApiKey = lab.type === 'unified';

    return this.request(url, 'GET', undefined, useXApiKey);
  }

  // Get thresholds/constants for a lab
  async getThresholds(labKey: string): Promise<any> {
    const lab = LABS[labKey];
    if (!lab) throw new Error(`Unknown lab: ${labKey}`);

    const url = this.getLabUrl(lab, '/thresholds');
    const useXApiKey = lab.type === 'unified';

    return this.request(url, 'GET', undefined, useXApiKey);
  }

  // Perform assessment/analysis
  async assess(labKey: string, data: any): Promise<any> {
    const lab = LABS[labKey];
    if (!lab) throw new Error(`Unknown lab: ${labKey}`);

    const url = this.getLabUrl(lab, lab.endpoint);
    const useXApiKey = lab.type === 'unified';

    return this.request(url, 'POST', data, useXApiKey);
  }

  // Get all labs from Unified API
  async getUnifiedLabs(): Promise<any> {
    return this.request(`${this.unifiedApiUrl}/labs`, 'GET', undefined, true);
  }

  // Get all labs status
  async getAllLabsStatus(): Promise<Record<string, { status: string; healthy: boolean }>> {
    const results: Record<string, { status: string; healthy: boolean }> = {};

    await Promise.allSettled(
      Object.keys(LABS).map(async (labKey) => {
        try {
          const health = await this.checkHealth(labKey);
          results[labKey] = { status: health.status || 'healthy', healthy: true };
        } catch (error) {
          results[labKey] = { status: 'offline', healthy: false };
        }
      })
    );

    return results;
  }

  // Validation status (Unified API)
  async getValidationStatus(): Promise<any> {
    return this.request(`${this.unifiedApiUrl}/validation/status`, 'GET', undefined, true);
  }

  // WebSocket connection for real-time updates
  createWebSocket(labName: string): WebSocket {
    const wsUrl = this.unifiedApiUrl.replace('http', 'ws');
    return new WebSocket(`${wsUrl}/ws/${labName}`);
  }

  // === MCP HTTP METHODS (Primary Agent/Tool Gateway) ===

  // Get featured tools (defaults to materials_rd department)
  async getMCPFeatured(department?: 'materials_rd' | 'life_sciences'): Promise<any> {
    const url = department
      ? `${this.mcpApiUrl}/featured?department=${department}`
      : `${this.mcpApiUrl}/featured`;

    const headers: HeadersInit = { 'Content-Type': 'application/json' };
    if (this.mcpApiKey) {
      headers['Authorization'] = `Bearer ${this.mcpApiKey}`;
    }

    const response = await fetch(url, { headers });
    if (!response.ok) throw new Error(`MCP Error: ${response.status}`);
    return response.json();
  }

  // Get all MCP tools (with optional department filter)
  async getMCPTools(department?: 'materials_rd' | 'life_sciences' | 'general'): Promise<any> {
    const url = department
      ? `${this.mcpApiUrl}/tools?department=${department}`
      : `${this.mcpApiUrl}/tools`;

    const headers: HeadersInit = { 'Content-Type': 'application/json' };
    if (this.mcpApiKey) {
      headers['Authorization'] = `Bearer ${this.mcpApiKey}`;
    }

    const response = await fetch(url, { headers });
    if (!response.ok) throw new Error(`MCP Error: ${response.status}`);
    return response.json();
  }

  // Call an MCP tool by name
  async callMCPTool(tool: string, params: Record<string, any>): Promise<any> {
    const headers: HeadersInit = { 'Content-Type': 'application/json' };
    if (this.mcpApiKey) {
      headers['Authorization'] = `Bearer ${this.mcpApiKey}`;
    }

    const response = await fetch(`${this.mcpApiUrl}/tools/call`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ tool, params })
    });

    if (!response.ok) throw new Error(`MCP Tool Error: ${response.status}`);
    return response.json();
  }

  // Get MCP lab cartography map
  async getMCPMap(): Promise<any> {
    const headers: HeadersInit = { 'Content-Type': 'application/json' };
    if (this.mcpApiKey) {
      headers['Authorization'] = `Bearer ${this.mcpApiKey}`;
    }

    const response = await fetch(`${this.mcpApiUrl}/map`, { headers });
    if (!response.ok) throw new Error(`MCP Error: ${response.status}`);
    return response.json();
  }

  // MCP health check
  async checkMCPHealth(): Promise<any> {
    const headers: HeadersInit = { 'Content-Type': 'application/json' };
    if (this.mcpApiKey) {
      headers['Authorization'] = `Bearer ${this.mcpApiKey}`;
    }

    try {
      const response = await fetch(`${this.mcpApiUrl}/health`, { headers });
      if (!response.ok) throw new Error(`MCP Error: ${response.status}`);
      return response.json();
    } catch (error) {
      return { status: 'offline', error: String(error) };
    }
  }

  // === MATERIALS & SYNTHESIS METHODS ===

  // Search synthesis archive (via MCP pocket tools)
  async searchSynthesisRecords(query: string, limit: number = 10): Promise<any> {
    return this.callMCPTool('pocket.search_lab_notes', { query, limit });
  }

  // Get synthesis history
  async getSynthesisHistory(limit: number = 20): Promise<any> {
    return this.callMCPTool('pocket.list_lab_notes', { limit, context: 'synthesis' });
  }

  // Add synthesis record
  async addSynthesisRecord(note: string, tags?: string[], priority?: string): Promise<any> {
    return this.callMCPTool('pocket.add_lab_note', {
      note,
      tags,
      context: 'synthesis',
      priority
    });
  }

  // Get material properties (via MCP materials tools)
  async getMaterialProperties(mpId: string): Promise<any> {
    return this.callMCPTool('materials.get_mp_material', { mp_id: mpId });
  }

  // Analyze material structure
  async analyzeMaterialStructure(filePath: string, citations: boolean = true): Promise<any> {
    return this.callMCPTool('materials.analyze_structure', { file_path: filePath, citations });
  }

  // Batch analyze materials
  async batchAnalyzeMaterials(filePaths: string[]): Promise<any> {
    return this.callMCPTool('materials.batch_analyze_structures', { file_paths: filePaths });
  }

  // Validate material structure
  async validateMaterialStructure(filePath: string): Promise<any> {
    return this.callMCPTool('materials.validate_structure', { file_path: filePath });
  }

  // Get materials database info
  async getMaterialsDatabaseInfo(): Promise<any> {
    return this.callMCPTool('materials.database_info', {});
  }

  // === AGENT & TELEMETRY METHODS ===

  // Get global system status (placeholder - extend as backend provides)
  async getGlobalSystemStatus(): Promise<any> {
    // This could call a unified endpoint or aggregate multiple lab healths
    try {
      const [mcpHealth, unifiedHealth, labsStatus] = await Promise.all([
        this.checkMCPHealth(),
        this.request(`${this.unifiedApiUrl}/health`, 'GET', undefined, true).catch(() => ({ status: 'offline' })),
        this.getAllLabsStatus()
      ]);

      return {
        mcp: mcpHealth,
        unified: unifiedHealth,
        labs: labsStatus,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      return { status: 'error', error: String(error) };
    }
  }

  // Get agent telemetry (placeholder for future backend endpoint)
  async getAgentTelemetry(): Promise<any> {
    // TODO: Wire to actual agent telemetry endpoint when available
    return {
      agents: [
        { id: 'AGENT_084', status: 'nominal', load: 0.42 },
        { id: 'AGENT_112', status: 'active', load: 0.78 },
        { id: 'AGENT_004', status: 'degraded', load: 0.91 }
      ],
      neural_load: { cognitive_overhead: 0.67, lattice_stress: 0.54 },
      mesh_topology: { active_nodes: 220, total_nodes: 250 }
    };
  }
}

export const apiClient = new APIClient();
