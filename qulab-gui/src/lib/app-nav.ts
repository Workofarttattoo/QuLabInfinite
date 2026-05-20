import { LABS } from './api-client';
import { getLabRoute } from './lab-routes';

/** Port (8001–8010) → lab config key. */
export const PORT_TO_LAB_KEY: Record<number, string> = Object.fromEntries(
  Object.entries(LABS)
    .filter(([, cfg]) => 'port' in cfg && typeof cfg.port === 'number')
    .map(([key, cfg]) => [(cfg as { port: number }).port, key]),
) as Record<number, string>;

export function getLabRouteByPort(port: number): string {
  const key = PORT_TO_LAB_KEY[port];
  return key ? getLabRoute(key) : '/labs';
}

/** Primary header nav (Product Hunt flow). */
export const APP_PRIMARY_NAV = [
  { label: 'Mission', path: '/' },
  { label: 'Labs', path: '/labs' },
  { label: 'Screens', path: '/screens' },
  { label: 'Medical', path: '/medical-directory' },
  { label: 'Dashboard', path: '/dashboard-os' },
  { label: 'Echo', path: '/echo-mission' },
] as const;

/** OS-style bottom bar used on Stitch / lab OS pages. */
export const APP_BOTTOM_NAV_OS = [
  { label: 'Dashboard', path: '/dashboard-os', icon: 'grid_view' },
  { label: 'Units', path: '/labs', icon: 'science' },
  { label: 'Mission', path: '/echo-mission', icon: 'assignment_late' },
  { label: 'System', path: '/system-lockdown', icon: 'settings_input_component' },
] as const;

/** Footer / secondary links on tactical pages. */
export const APP_FOOTER_LINKS = [
  { label: 'Labs', path: '/labs' },
  { label: 'Echo MCP', path: '/echo/integrations' },
  { label: 'Intel', path: '/intel-dashboard' },
  { label: 'Screens', path: '/screens' },
] as const;
