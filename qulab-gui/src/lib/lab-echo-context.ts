import { LABS } from './api-client';
import { LAB_ROUTE_BY_KEY } from './lab-routes';

/** URL slug (path segment after /labs/) → LABS config key */
export const LAB_KEY_BY_SLUG: Record<string, string> = Object.fromEntries(
  Object.entries(LAB_ROUTE_BY_KEY).map(([key, path]) => [path.replace(/^\/labs\//, ''), key])
);

export interface LabEchoContext {
  pathname: string;
  labSlug: string;
  labKey?: string;
  labName?: string;
  category?: string;
}

export function resolveLabEchoContext(pathname: string): LabEchoContext | null {
  const labsMatch = pathname.match(/^\/labs\/([^/]+)/);
  if (labsMatch) {
    const labSlug = labsMatch[1];
    const labKey = LAB_KEY_BY_SLUG[labSlug] ?? labSlug;
    const config = LABS[labKey];
    return {
      pathname,
      labSlug,
      labKey,
      labName: config?.name ?? labSlug.replace(/-/g, ' '),
      category: config?.category,
    };
  }

  const hubRoutes: Record<string, Partial<LabEchoContext>> = {
    '/': { labSlug: 'mission', labName: 'Mission Control' },
    '/screens': { labSlug: 'screens', labName: 'Stitch Screens' },
    '/medical-directory': { labSlug: 'medical', labName: 'Medical Directory', category: 'medical' },
    '/dashboard-os': { labSlug: 'dashboard-os', labName: 'Global Dashboard OS' },
    '/dashboard': { labSlug: 'dashboard', labName: 'Global Dashboard' },
    '/materials-production': { labSlug: 'materials-production', labName: 'Materials Production', category: 'materials' },
    '/synthesis-archive': { labSlug: 'synthesis-archive', labName: 'Synthesis Archive', category: 'materials' },
    '/echo': { labSlug: 'echo', labName: 'Echo Control Center' },
    '/echo-mission': { labSlug: 'echo-mission', labName: 'Echo Mission Foyer' },
    '/intel-dashboard': { labSlug: 'intel', labName: 'Global Intel Dashboard' },
    '/hive-mind': { labSlug: 'hive-mind', labName: 'Hive Mind Intel Mesh' },
    '/hive-mind-dashboard': { labSlug: 'hive-mind', labName: 'Hive Mind Dashboard' },
    '/agent-telemetry': { labSlug: 'telemetry', labName: 'Agent Telemetry' },
    '/system-lockdown': { labSlug: 'system', labName: 'System Lockdown' },
    '/labs-fleet': { labSlug: 'fleet', labName: 'Live Lab Fleet' },
    '/labs-old': { labSlug: 'fleet', labName: 'Lab Fleet (Legacy)' },
  };

  const echoAdminMatch = pathname.match(/^\/echo\/([^/]+)/);
  if (echoAdminMatch) {
    const section = echoAdminMatch[1];
    const echoNames: Record<string, string> = {
      workload: 'Echo Workload',
      training: 'Echo Training',
      settings: 'Echo Settings',
      integrations: 'Echo MCP Integrations',
    };
    return {
      pathname,
      labSlug: `echo-${section}`,
      labName: echoNames[section] ?? `Echo ${section}`,
    };
  }

  const hub = hubRoutes[pathname];
  if (hub) {
    return { pathname, labSlug: hub.labSlug ?? 'hub', ...hub };
  }

  if (pathname === '/labs') {
    return { pathname, labSlug: 'fleet', labName: 'Lab Fleet' };
  }

  return null;
}

/**
 * Fixed bottom Echo dock — hidden only on `/echo` control center (it has its own console).
 * `/echo-mission` and `/echo/*` sub-routes use the shared dock.
 */
export function shouldShowEchoCommandBar(pathname: string): boolean {
  return pathname !== '/echo';
}

/** In-page Echo panel — hidden when the dock is visible to avoid duplicate inputs. */
export function shouldShowEchoCommandInline(pathname: string): boolean {
  return !shouldShowEchoCommandBar(pathname);
}
