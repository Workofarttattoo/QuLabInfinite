# Materials & R&D gateway — production checklist

The **unified MCP HTTP server** (`unified_mcp_server.py`) is the primary agent-facing surface for **materials, chemistry, physics, Ech0 invention tooling, and lab-floor (Flash Joule) workflows**. Use this checklist when deploying.

## Required vs optional

| Concern | Strict production | Dev / staging without full data |
|--------|-------------------|----------------------------------|
| Materials JSONL (`materials.get_mp_material`) | Mount file; set `QU_LAB_MATERIALS_JSONL` | Set `QU_LAB_MATERIALS_DATASET_OPTIONAL=true` — server starts; MP tool returns **503** until data exists |
| API authentication | Set `QULAB_MCP_API_KEY` — clients send `Authorization: Bearer …` or `X-MCP-Api-Key` | Leave unset (open), only on trusted networks |
| Usage audit | Set `QULAB_USAGE_EVENTS_PATH=/path/to/events.jsonl` | Omit |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `QU_LAB_MATERIALS_JSONL` | Absolute or relative path to Materials Project expansion JSONL |
| `QU_LAB_MATERIALS_DATASET_OPTIONAL` | `true` / `1` — do not fail startup if JSONL missing |
| `QULAB_MCP_API_KEY` | If set, all routes except `/health` require bearer or `X-MCP-Api-Key` |
| `QULAB_USAGE_EVENTS_PATH` | Append request/tool JSONL events for oversight |
| `QU_LAB_MCP_HOST` | Bind address (default `0.0.0.0`) |
| `QU_LAB_MCP_PORT` | Listen port (default **8102**) |

## HTTP routes (materials/R&D first)

| Method | Path | Use |
|--------|------|-----|
| GET | `/health` | Probe; includes `materials_dataset`, `tools_by_department`, `materials_mp_ready` |
| GET | `/featured?department=materials_rd` | **Default product home** — curated list of tools for UI and agents |
| GET | `/tools` | Full catalog |
| GET | `/tools?department=materials_rd` | Same tools as featured, with full stats block |
| POST | `/tools/call` | Body: `{"tool": "materials.analyze_structure", "params": {"file_path": "..."}}` |
| GET | `/map` | Cartography + dataset summary |

### Department labels

Each tool exposes `department` in JSON:

- **`materials_rd`** — curated materials, chemistry, physics, Ech0, pocket/FJH, calculator; plus dynamic labs whose cartographer `domain` is chemistry, physics, nanotechnology, engineering, mathematics, or computation.
- **`life_sciences`** — dynamic labs tagged biology/medicine.
- **`general`** — everything else.

## New curated tool

- **`materials.validate_structure`** — fast CIF/POSCAR validation (no full provenance build).

## Unified REST API (separate process)

`api/unified_api.py` uses **`QU_LAB_MASTER_KEYS`** and serves REST/WebSocket on port **8000** by default. It is a **different** app from the MCP server; product teams often use **MCP (8102)** for agents and **Unified API** for browser REST. See `docs/FIGMA_BACKEND_WIRING.md`.

## Smoke test

```bash
# From repo root, with venv + deps
python unified_mcp_server.py &
curl -s localhost:8102/health | python -m json.tool
curl -s "localhost:8102/featured" | python -m json.tool
```

With auth:

```bash
export QULAB_MCP_API_KEY='your-secret'
curl -s -H "Authorization: Bearer your-secret" localhost:8102/featured
```
