import { apiClient } from './api-client';
import type { LabEchoContext } from './lab-echo-context';

export interface EchoCommandResult {
  ok: boolean;
  summary: string;
  detail?: string;
  tool?: string;
}

export interface EchoCommandOptions {
  context?: LabEchoContext | null;
}

function formatDetail(value: unknown): string | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'string') return value.slice(0, 500);
  try {
    return JSON.stringify(value, null, 0).slice(0, 500);
  } catch {
    return String(value).slice(0, 500);
  }
}

function extractMaterialName(text: string): string {
  const analyzeMatch = text.match(/(?:analyze|search|find|lookup)\s+(.+)/i);
  if (analyzeMatch) return analyzeMatch[1].trim();
  const quoted = text.match(/["']([^"']+)["']/);
  if (quoted) return quoted[1];
  return text.trim();
}

function appendLabNote(text: string, context?: LabEchoContext | null) {
  const key = 'qulab_echo_command_log';
  try {
    const prev = JSON.parse(sessionStorage.getItem(key) ?? '[]') as Array<Record<string, string>>;
    prev.unshift({
      at: new Date().toISOString(),
      lab: context?.labName ?? context?.labSlug ?? 'unknown',
      text,
    });
    sessionStorage.setItem(key, JSON.stringify(prev.slice(0, 50)));
  } catch {
    /* ignore quota */
  }
}

async function invokeTool(
  tool: string,
  params: Record<string, unknown>,
  summary: string
): Promise<EchoCommandResult> {
  const payload = await apiClient.callMCPTool(tool, params);
  const result = (payload as { result?: unknown })?.result ?? payload;
  return {
    ok: true,
    summary,
    detail: formatDetail(result),
    tool,
  };
}

/** Route natural-language Echo commands to onboard ECH0 / MCP tools for the active lab. */
export async function executeEchoCommand(
  command: string,
  options: EchoCommandOptions = {}
): Promise<EchoCommandResult> {
  const text = command.trim();
  const ctx = options.context ?? null;

  if (!text) {
    return { ok: false, summary: 'Type an instruction for Echo first.' };
  }

  appendLabNote(text, ctx);
  const lower = text.toLowerCase();
  const labKey = ctx?.labKey ?? '';
  const category = ctx?.category ?? '';

  try {
    if (/^(help|\?|tools|list tools)$/.test(lower) || lower.startsWith('help ')) {
      const payload = await apiClient.getMCPTools();
      const names: string[] = Array.isArray(payload?.tools)
        ? payload.tools.map((t: { name?: string }) => t.name ?? String(t))
        : [];
      const ech0 = names.filter((n) => n.startsWith('ech0.'));
      return {
        ok: true,
        summary: `${names.length} MCP tools · ${ech0.length} ECH0 tools`,
        detail: [
          'Try: analyze graphene',
          'database info',
          'recommend for aerospace',
          'invent: solar coating',
          'status',
        ].join(' · '),
        tool: 'GET /tools',
      };
    }

    if (/^(status|health|ping|labs|system)$/.test(lower) || lower.startsWith('status ')) {
      const status = await apiClient.getGlobalSystemStatus();
      const mcpStatus = status?.mcp?.status ?? 'unknown';
      const mcpOnline = mcpStatus !== 'offline' && !status?.mcp?.error;
      const labs = (status?.labs ?? {}) as Record<string, { healthy?: boolean }>;
      const online = Object.values(labs).filter((l) => l.healthy).length;
      const total = Object.keys(labs).length;

      return {
        ok: mcpOnline,
        summary: mcpOnline
          ? `MCP ${mcpStatus} · ${online}/${total} labs reachable`
          : 'MCP offline on :8102',
        detail: mcpOnline
          ? ctx?.labName
            ? `Active screen: ${ctx.labName}`
            : undefined
          : 'Run: PYTHONPATH=. python3 unified_mcp_server.py',
        tool: 'system.status',
      };
    }

    if (
      /\b(database|db info|how many materials|material count)\b/i.test(text) ||
      lower === 'database'
    ) {
      return invokeTool('materials.database_info', {}, 'Materials database metadata');
    }

    const mpId = text.match(/\bmp-\d+\b/i)?.[0];
    if (mpId) {
      return invokeTool(
        'materials.get_mp_material',
        { mp_id: mpId.toLowerCase() },
        `Loaded ${mpId}`
      );
    }

    if (/\b(invent|invention|prototype)\b/i.test(text)) {
      const colon = text.split(/[:—-]\s*/, 2);
      const name = (colon[1] ?? colon[0]).slice(0, 80).trim() || 'Lab invention';
      const description = text;
      return invokeTool(
        'ech0.quick_invention',
        {
          name,
          description,
          application: category === 'medical' ? 'medical' : 'general',
          budget: 500,
        },
        'ECH0 invention run started'
      );
    }

    if (/\b(recommend|design|select|choose)\b/i.test(text)) {
      const forMatch = text.match(/(?:for|application)\s+(.+)/i);
      const application = forMatch?.[1]?.trim() || 'structural';
      const budgetMatch = text.match(/\$?(\d+(?:\.\d+)?)\s*(?:\/kg|per kg)?/i);
      const budget = budgetMatch ? Number(budgetMatch[1]) : 100;
      return invokeTool(
        'ech0.design_selector',
        { application, budget_per_kg: budget },
        `Material recommendations for ${application}`
      );
    }

    if (/\b(calc|calculate|compute)\b/i.test(text)) {
      const exprMatch = text.match(/(?:calc|calculate|compute)\s+(.+)/i);
      const expr = exprMatch?.[1]?.trim() ?? text.replace(/^[\d\s+\-*/().]+$/, text);
      return invokeTool('ai.calc', { expr }, 'Calculation result');
    }

    const smilesMatch = text.match(/(?:smiles|validate)\s+([A-Za-z0-9@+\-\[\]()=#$\\/\\.]+)/i);
    if (smilesMatch || (category === 'materials' && /^[A-Za-z0-9@+\-\[\]()=#$\\/\\.]{4,}$/.test(text))) {
      const smiles = smilesMatch?.[1] ?? text.trim();
      return invokeTool('chemistry.validate_smiles', { smiles }, 'SMILES validation');
    }

    if (
      labKey === 'materials' ||
      category === 'materials' ||
      /\b(analyze|analysis|material|graphene|steel|polymer|ceramic)\b/i.test(text)
    ) {
      const materialName = extractMaterialName(text);
      return invokeTool(
        'ech0.analyze_material',
        { material_name: materialName },
        `ECH0 analyzed “${materialName}”`
      );
    }

    if (labKey === 'chemistry' || ctx?.labSlug?.includes('chem')) {
      const smiles = text.trim();
      if (smiles.length >= 3) {
        return invokeTool(
          'chemistry.analyze_molecule',
          { smiles, citations: true },
          'Molecule analysis submitted'
        );
      }
    }

    // Default: ECH0 interprets instruction in current lab context
    const materialName = extractMaterialName(text);
    const result = await invokeTool(
      'ech0.analyze_material',
      { material_name: materialName },
      ctx?.labName
        ? `Echo handled instruction in ${ctx.labName}`
        : 'Echo processed your instruction'
    );
    return result;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const needsMcp =
      message.includes('Failed to fetch') ||
      message.includes('MCP') ||
      message.includes('8102') ||
      message.includes('404');

    return {
      ok: false,
      summary: 'Echo could not reach the lab runtime',
      detail: needsMcp
        ? 'Start MCP: PYTHONPATH=. python3 unified_mcp_server.py (port 8102)'
        : message,
    };
  }
}
