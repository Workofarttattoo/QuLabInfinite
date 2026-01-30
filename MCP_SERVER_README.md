# Master MCP Server (QuLabInfinite)

## Start
```bash
export MCP_API_KEY="set-me"            # optional
export MCP_HOST=0.0.0.0                # optional
export MCP_PORT=8000                   # optional
python master_mcp_server.py
```

## Endpoints
- `GET /health` – status check
- `GET /tools` – list tool metadata (cost, lite availability, schema)
- `POST /call` – invoke a tool with JSON body:
```json
{"tool": "ai.calc", "args": {"expr": "2+2"}, "lite": false}
```

## Client helpers
```bash
python scripts/master_mcp_client.py              # lists tools and demo calls
python scripts/mcp_smoke_test.py                 # minimal smoke checks
python scripts/agent_eval.py                     # lightweight eval assertions
```

Env vars for clients:
- `MCP_API_URL` (default `http://localhost:8000`)
- `MCP_API_KEY` (optional; must match server)

## Data prep for fine-tuning
```bash
python scripts/agent_data_prep.py --out training_data/agent_ft_dataset.jsonl
```
Skips missing sources; aggregates JSON/JSONL records where available.
