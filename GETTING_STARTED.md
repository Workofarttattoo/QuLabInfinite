# Getting Started with QuLab Infinite

Welcome! This guide will get you up and running in **5 minutes**.

---

## System Requirements

- **Python 3.8+** (3.11 recommended)
- **pip** or **conda**
- **8GB+ RAM** (medical labs benefit from 16GB+)
- **macOS, Linux, or Windows (WSL)**

---

## Step 1: Clone & Install (2 minutes)

```bash
# Clone the repository
git clone https://github.com/Workofarttattoo/QuLabInfinite.git
cd QuLabInfinite

# Install QuLab
pip install -e .

# Optional: Create a virtual environment first
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Verify Installation
```bash
python -c "import qulab; print('✓ QuLab installed')"
```

---

## Step 2: Configure (1 minute)

Copy the example environment file and edit if needed:

```bash
cp .env.secure.example .env
```

Edit `.env` with your settings (optional for demo):
```bash
# Generate a strong API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env:
QU_LAB_MASTER_KEYS=your-generated-key-here
QU_LAB_MCP_PORT=8102
```

---

## Step 3: Launch (Choose One)

### Option A: Full Stack (All three gateways + GUI) ⭐ **Recommended**
```bash
bash LAUNCH_PRODUCT_HUNT.sh
```

This starts:
- 🔵 **MCP HTTP** on `localhost:8102` (agents, tool orchestration)
- 🟢 **Unified REST API** on `localhost:8000` (REST, WebSocket)
- 🟡 **Medical Labs** on `localhost:8001-8010` (10 diagnostic systems)
- 💻 **Web GUI** on `localhost:3000` (beautiful Figma-designed interface)

Then opens: **http://localhost:3000** automatically

### Option B: MCP Only (Recommended for first demo)
```bash
bash LAUNCH_PRODUCT_HUNT.sh mcp-only
# Or directly:
python unified_mcp_server.py
```

Then open: **http://localhost:8102/featured**

### Option C: REST API Only
```bash
bash LAUNCH_PRODUCT_HUNT.sh rest-only
# Or directly:
uvicorn api.unified_api:app --reload
```

Then open: **http://localhost:8000/docs**

### Option D: Medical Labs Only
```bash
bash LAUNCH_PRODUCT_HUNT.sh medical-only
```

Then open: **http://localhost:8001/docs** (Alzheimer's), etc.

---

## Step 4: Verify It's Working

### Test MCP Server
```bash
curl http://localhost:8102/health
# Expected: {"status": "healthy", "timestamp": "..."}

curl http://localhost:8102/featured
# Expected: List of 50+ curated tools
```

### Test REST API
```bash
curl http://localhost:8000/docs
# Expected: Interactive Swagger UI opens in browser
```

### Test Medical Labs
```bash
curl http://localhost:8001/docs
# Expected: Alzheimer's Early Detection API docs
```

---

## Step 5: Try Your First Query

### Via MCP (Tool-based)
```bash
curl -X POST http://localhost:8102/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "ai.calc",
    "params": {"expr": "sqrt(16) + 2**3"}
  }'
# Expected: {"result": 12.0}
```

### Via REST API
```bash
curl -X POST http://localhost:8000/quantum/simulate \
  -H "X-Api-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "system_type": "transmon",
    "num_qubits": 3,
    "circuit_depth": 5,
    "algorithm": "vqe"
  }'
```

### Via Medical Lab
```bash
curl -X POST http://localhost:8001/assess \
  -H "Content-Type: application/json" \
  -d '{
    "age": 72,
    "apoe_e4_copies": 1,
    "mri_hippocampal_volume_ml": 2500,
    "amyloid_pet_suvr": 1.35,
    "tau_pet_suvr": 1.25,
    "csf_phospho_tau_pg_ml": 65
  }'
# Returns: {"diagnosis": "MCI due to AD", "confidence": 0.92, "risk_5y": 0.67}
```

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────┐
│          Client (Browser, Agent, CLI)           │
└────────────┬────────────┬──────────────┬────────┘
             │            │              │
        ┌────▼──┐    ┌────▼──┐    ┌─────▼─┐
        │ MCP   │    │ REST  │    │Medical│
        │ 8102  │    │ 8000  │    │8001-10│
        └────┬──┘    └────┬──┘    └─────┬─┘
             │            │             │
    ┌────────▼──────────┬─▼────────┬────▼─────┐
    │  Materials Lab    │ Quantum  │ Chemistry│
    │  Medical Diagn.   │ Physics  │ Biology  │
    └───────────────────┴──────────┴──────────┘
```

**Three independent gateways, same underlying labs.**

---

## Next Steps

### For Agents/LLMs
1. Read: [docs/FIGMA_BACKEND_WIRING.md](docs/FIGMA_BACKEND_WIRING.md)
2. Learn: [MCP specification](https://modelcontextprotocol.io/)
3. Use: `GET /tools` to discover all available tools
4. Call: `POST /tools/call` with `{"tool": "...", "params": {...}}`

### For Frontend/UI Teams
1. Read: [docs/FIGMA_BACKEND_WIRING.md](docs/FIGMA_BACKEND_WIRING.md)
2. Import Figma design: [QuLab Console v1](https://www.figma.com/design/N9joP1YMYdWU1kWWIZbTBm)
3. Wire components to REST API routes in `api.unified_api.py`
4. Annotate with proof artifacts from `/reports/golden_runs/`

### For Research Labs
1. Start with Medical labs (`bash LAUNCH_PRODUCT_HUNT.sh medical-only`)
2. Explore a specific diagnostic (Alzheimer's on port 8001)
3. Try the `/assess` endpoint with your patient data
4. Review clinical accuracy & validation standards in lab documentation

### For Product Builders
1. Read: [PRODUCT_HUNT.md](PRODUCT_HUNT.md) (60-sec pitch & differentiators)
2. Clone & launch
3. Monitor logs: `tail -f logs/*.log`
4. Contribute to roadmap: [GitHub Issues](https://github.com/Workofarttattoo/QuLabInfinite/issues)

---

## Troubleshooting

### Port Already in Use
```bash
# Kill existing process on port 8102 (example)
lsof -ti:8102 | xargs kill -9

# Or use a different port
QU_LAB_MCP_PORT=9102 python unified_mcp_server.py
```

### Python Module Not Found
```bash
# Ensure you installed in editable mode
pip install -e .

# Or add repo to PYTHONPATH
export PYTHONPATH="${PWD}:$PYTHONPATH"
```

### API Key Errors
```bash
# Check .env is properly loaded
python -c "from core.security import load_api_keys_from_env; print(load_api_keys_from_env())"

# If empty, generate a key
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Add to .env: QU_LAB_MASTER_KEYS=<the-key>
```

### Medical Labs Not Starting
```bash
# Check if scripts/start_medical_labs.sh exists
ls scripts/start_medical_labs.sh

# Run with explicit environment
LAB_HOST=0.0.0.0 LAB_PORT_PREFIX=800 bash scripts/start_medical_labs.sh
```

---

## Stopping Services

```bash
# Stop all gateways
bash STOP_QULAB.sh

# Or manually
pkill -f "unified_mcp_server.py"
pkill -f "uvicorn"
```

---

## Performance Tips

### For Medical Labs (large simulations)
- Allocate 16GB+ RAM
- Use SSD for data cache
- Run on 4+ CPU cores

### For MCP Server (many agents)
- Monitor `/health` endpoint for latency
- Scale horizontally with load balancer (Nginx, HAProxy)
- Consider caching frequent queries

### For REST API (browser traffic)
- Enable CORS (configured in `api.unified_api.py`)
- Use WebSocket for real-time updates
- Cache results in Redis (optional)

---

## Environment Variables (Complete Reference)

```bash
# API Keys & Security
QU_LAB_MASTER_KEYS=key1,key2,key3              # Comma-separated API keys
QULAB_MCP_API_KEY=optional-mcp-specific-key    # MCP-only key
JWT_SECRET_KEY=your-jwt-secret                  # For token generation

# Server Configuration
QU_LAB_MCP_PORT=8102                           # MCP HTTP port
QU_LAB_HOST=0.0.0.0                            # Bind address
QU_LAB_MASTER_TIER=enterprise                  # Free|standard|enterprise

# Rate Limiting
QU_LAB_MASTER_RATE_LIMIT=1000                  # Requests per minute

# Optional: External APIs
MATERIALS_PROJECT_API_KEY=your-mp-key          # Materials Project DB
TWILIO_ACCOUNT_SID=your-twilio-sid             # For notifications
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_PHONE_NUMBER=+1234567890

# Optional: Data Paths
QU_LAB_MATERIALS_JSONL=/path/to/custom/data    # Custom materials dataset
```

---

## Common Commands

```bash
# Launch everything
bash LAUNCH_PRODUCT_HUNT.sh

# Stop everything
bash STOP_QULAB.sh

# View logs
tail -f logs/MCP-Server.log
tail -f logs/Unified-API.log
tail -f logs/medical-labs.log

# Test MCP server
curl http://localhost:8102/featured | jq '.'

# Test REST API with auth
curl http://localhost:8000/labs \
  -H "X-Api-Key: your-api-key"

# Interactive Python shell
python
>>> from materials_lab.materials_lab import MaterialsLab
>>> lab = MaterialsLab()
>>> lab.search_database("Si")
```

---

## Next: Product Hunt Launch

You're now ready to demo QuLab to the Product Hunt community!

- **Pitch:** See [PRODUCT_HUNT.md](PRODUCT_HUNT.md)
- **Frontend wiring:** See [docs/FIGMA_BACKEND_WIRING.md](docs/FIGMA_BACKEND_WIRING.md)
- **Production guide:** See [docs/MATERIALS_RD_PRODUCTION.md](docs/MATERIALS_RD_PRODUCTION.md)

---

## Support

- **Questions?** Check [docs/](docs/) or GitHub Issues
- **Bug reports?** [GitHub Issues](https://github.com/Workofarttattoo/QuLabInfinite/issues)
- **Email:** [support@aios.is](mailto:support@aios.is)

---

**Happy scientific computing! 🚀**

