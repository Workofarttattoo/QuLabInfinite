# Figma Make — QuLab Infinite GUI (canonical UI)

**Project name:** **QuLab Infinite GUI** (Make title: *Connect Frontend to Backend*).  
**This is the primary UI project** for Product Hunt — a **Figma Make** file (live React + routes), not the separate Design file `QuLab Infinite — Lab Console UX v1`.

**Published site:** [https://pound-goblin-20062908.figma.site](https://pound-goblin-20062908.figma.site) (boot → `/labs` → individual lab screens).

## Link

**[Connect Frontend to Backend](https://www.figma.com/make/j9P7lJtC6OdLpVY0neIaRt/Connect-Frontend-to-Backend)**

- **Make file key:** `j9P7lJtC6OdLpVY0neIaRt`
- **Type:** Figma Make (`/make/…`) — exports runnable React + routes

## What’s inside

- **Boot:** `BootSequence.tsx` (Three.js particle boot → Mission Control)
- **Hub:** `MissionControl`, `GlobalDashboard`, `GlobalDashboardOS`, `MedicalDirectoryUnified`
- **Labs:** 40+ pages (materials, chemistry, medical 8001–8010, Echo, telemetry, etc.)
- **Backend client:** `src/lib/api-client.ts` — MCP `:8102`, Unified API `:8000`, medical `:8001–8010`
- **Routes:** `src/app/routes.tsx`

## Local repo copy

Synced into **`qulab-gui/`** for `npm run dev` / Product Hunt launch (see `qulab-gui/README.md`).

To re-sync after editing in Make, run:

```bash
# Re-run the Figma Make → qulab-gui sync (agent or manual export from Make UI)
```

Or in Figma Make: use **Export / Download code** and merge into `qulab-gui/`.

## Common preview error: `apiClient` export

If Make shows:

`The requested module '/src/lib/hooks.ts' does not provide an export named 'apiClient'`

**Fix in Make** — edit `src/lib/hooks.ts` and add after the imports:

```ts
export { apiClient, LABS } from './api-client';
```

Or import `apiClient` from `../../lib/api-client` in pages (see `qulab-gui/FIGMA_MAKE_FIX_hooks.ts.txt`). The synced `qulab-gui/` repo already includes this fix.

## MCP limitations

- **`use_figma` / `get_metadata`** do not work on Make files.
- **`get_design_context`** works and returns the **source file list** + code.

## Env (local dev)

```bash
VITE_MCP_API_URL=/mcp              # Vite proxy → localhost:8102
VITE_UNIFIED_API_URL=http://localhost:8000
VITE_MEDICAL_BASE_URL=http://localhost
# VITE_API_KEY= / VITE_MCP_API_KEY= if gateways require auth
```

## Wrong file (do not use for this effort)

Design-only scratch file: [Lab Console UX v1](https://www.figma.com/design/N9joP1YMYdWU1kWWIZbTBm) — early wireframes, not the Make app.
