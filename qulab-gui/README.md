# QuLab Lab Console (React)

Product Hunt–ready GUI for the **MCP HTTP gateway** (`POST /tools/call` on port **8102**).

## Quick start

**Port 3000 on this machine is often Grafana (Docker)** — use **5173** for the lab GUI.

```bash
# Terminal 1 — MCP runtime
cd /path/to/QuLabInfinite
PYTHONPATH=. python3 unified_mcp_server.py

# Terminal 2 — GUI (must stay running; "can't open site" = server not started)
cd /path/to/QuLabInfinite
bash scripts/start-qulab-gui.sh
# or: cd qulab-gui && npm install && npm run dev
```

Open **http://127.0.0.1:5173/labs/materials** (use `127.0.0.1`, not `localhost:3000`)

If `npm install` fails on peer deps, `qulab-gui/.npmrc` already sets `legacy-peer-deps=true`.

## Production preview

```bash
npm run build
npm run preview -- --host 0.0.0.0 --port 3000
```

Or use **`bash LAUNCH_PRODUCT_HUNT.sh`** from the repo root (starts MCP, API, medical labs, and builds/serves the GUI).

## Echo command bar (every lab screen)

A fixed **Instruct Echo** bar appears on all `/labs/*` routes (and lab hubs). Commands route to onboard **ECH0** via MCP on port **8102**:

| You type | MCP tool |
|----------|----------|
| `analyze graphene` | `ech0.analyze_material` |
| `database info` | `materials.database_info` |
| `mp-1234567` | `materials.get_mp_material` |
| `recommend for aerospace` | `ech0.design_selector` |
| `invent: solar coating` | `ech0.quick_invention` |
| `status` / `help` | system / tool list |

Start the runtime: `PYTHONPATH=. python3 unified_mcp_server.py`

## Routes (mirror Figma frames)

| Route | Screen |
|-------|--------|
| `/` | Boot |
| `/command` | Command center |
| `/materials` | Materials bundle |
| `/chemistry` | Chemistry bundle |
| `/rd` | R&D orchestration |
| `/medical` | Medical directory (8001–8010) |
| `/unlock` | Materials + R&D synergy |

## Backend

- Dev: Vite proxies **`/mcp`** → `http://127.0.0.1:8102`
- Override: `VITE_MCP_BASE=https://your-host:8102`
- Optional auth: `VITE_MCP_API_KEY` when `QULAB_MCP_API_KEY` is set on the server

See **`docs/FIGMA_BACKEND_WIRING.md`** for full gateway matrix.
