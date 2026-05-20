/** Maps LABS config keys → React Router paths (must match routes.tsx). */
export const LAB_ROUTE_BY_KEY: Record<string, string> = {
  materials: '/labs/materials',
  quantum: '/labs/quantum',
  chemistry: '/labs/chemistry',
  oncology: '/labs/metabolic',
  drug: '/labs/drug',
  genomics: '/labs/genomics',
  immune: '/labs/genomics',
  metabolic: '/labs/metabolic',
  alzheimers: '/labs/alzheimers',
  parkinsons: '/labs/parkinsons',
  autoimmune: '/labs/autoimmune',
  sepsis: '/labs/sepsis',
  wound: '/labs/wound-healing',
  bone: '/labs/bone-density',
  kidney: '/labs/kidney-function',
  liver: '/labs/liver-disease',
  lung: '/labs/lung-function',
  pain: '/labs/pain-management',
};

export function getLabRoute(labKey: string): string {
  return LAB_ROUTE_BY_KEY[labKey] ?? `/labs/${labKey}`;
}

/** Stitch → React routes (hero Product Hunt screens). */
export const STITCH_HERO_SCREENS = [
  {
    id: 'global-dashboard',
    title: 'Global Dashboard',
    subtitle: 'Lab OS tactical overlay',
    path: '/dashboard-os',
    icon: 'public',
  },
  {
    id: 'medical-directory',
    title: 'Medical Directory',
    subtitle: 'Ports 8001–8010',
    path: '/medical-directory',
    icon: 'medical_services',
  },
  {
    id: 'materials-discovery',
    title: 'Materials Discovery',
    subtitle: 'Structures & MP database',
    path: '/labs/materials',
    icon: 'science',
  },
  {
    id: 'molecular-synthesis',
    title: 'Molecular Synthesis',
    subtitle: 'Chemistry & MD',
    path: '/labs/universal-chemistry',
    icon: 'biotech',
  },
  {
    id: 'metabolic-optimizer',
    title: 'Cancer Metabolic Optimizer',
    subtitle: 'Metabolic modeling',
    path: '/labs/metabolic',
    icon: 'monitor_heart',
  },
  {
    id: 'genomics-stem',
    title: 'Stem Cell / Genomics',
    subtitle: 'Genomics production',
    path: '/labs/genomics-production',
    icon: 'genetics',
  },
  {
    id: 'echo-mission',
    title: 'Echo Mission Foyer',
    subtitle: 'ECH0 command center',
    path: '/echo-mission',
    icon: 'hub',
  },
  {
    id: 'intel-dashboard',
    title: 'Global Intel',
    subtitle: 'Tactical intelligence mesh',
    path: '/intel-dashboard',
    icon: 'radar',
  },
  {
    id: 'labs-fleet',
    title: 'Live Lab Fleet',
    subtitle: 'All bootable units',
    path: '/labs-fleet',
    icon: 'view_module',
  },
] as const;
