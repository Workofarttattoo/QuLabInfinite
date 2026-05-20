# Product Hunt launch — QuLab GUI

## Start stack

```bash
# Terminal 1 — MCP gateway (ECH0 tools, port 8102)
cd /Users/noone/QuLabInfinite
PYTHONPATH=. python3 unified_mcp_server.py

# Terminal 2 — GUI (port 5173, avoids Grafana on :3000)
bash scripts/start-qulab-gui.sh
```

Open **http://127.0.0.1:5173**

## Demo flow

1. Boot sequence → Mission Control (`/`)
2. **INITIALIZE SYSTEM** or **Labs** → `/labs` (all labs by field)
3. **Screens** → `/screens` (Stitch hero hub)
4. Hero screens: Dashboard OS, Medical Directory, Materials, Metabolic, etc.
5. **Instruct Echo** — fixed bottom command bar on every route except `/echo` (control center has its own console). `/echo-mission` and `/echo/workload` etc. use the shared dock.

## Navigation map

| Label | Path |
|-------|------|
| Mission | `/` |
| Labs | `/labs` |
| Screens | `/screens` |
| Medical | `/medical-directory` |
| Dashboard | `/dashboard-os` |
| Echo | `/echo-mission` |

Legacy Figma paths redirect: `/units` → `/labs`, `/mission` → `/echo-mission`, `/system` → `/system-lockdown`.

## Published Figma site

Local fixes live in `qulab-gui/`. To update the public Figma site, re-sync from Make and republish.
