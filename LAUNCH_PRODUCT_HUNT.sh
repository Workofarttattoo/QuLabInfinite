#!/bin/bash
# QuLab Infinite Product Hunt Launch Script
# Start all three gateways in parallel for maximum impact

set -e

echo "🚀 QuLab Infinite — Product Hunt Launch"
echo "========================================"
echo ""
echo "Starting three main gateways..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for .env
if [ ! -f .env ]; then
    echo "${YELLOW}⚠️  .env not found. Creating from .env.secure.example...${NC}"
    if [ -f .env.secure.example ]; then
        cp .env.secure.example .env
        echo "${GREEN}✓ .env created. Edit with your API keys if needed.${NC}"
    else
        echo "${YELLOW}Note: Create a .env file with API keys for full functionality${NC}"
    fi
fi
echo ""

# Create logs directory
mkdir -p logs

# Function to print section header
print_header() {
    echo ""
    echo "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "${BLUE}$1${NC}"
    echo "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Function to start a gateway
start_gateway() {
    local name=$1
    local command=$2
    local port=$3
    local log_file="logs/${name}.log"

    echo "${GREEN}Starting $name on port $port...${NC}"
    eval "$command" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "logs/${name}.pid"
    sleep 2

    if kill -0 $pid 2>/dev/null; then
        echo "${GREEN}✓ $name started (PID: $pid)${NC}"
        echo "  Logs: $log_file"
        echo "  Base URL: http://localhost:$port"
    else
        echo "${YELLOW}✗ Failed to start $name. Check logs:${NC}"
        cat "$log_file"
        exit 1
    fi
}

# Option 1: Start just the primary MCP server
if [ "$1" = "mcp-only" ]; then
    print_header "MCP Server (Materials & R&D)"
    PY="${PYTHON:-python3}"
start_gateway "MCP-Server" "$PY unified_mcp_server.py" 8102
    echo ""
    echo "${GREEN}MCP Server running on http://localhost:8102${NC}"
    echo "  - Featured tools: http://localhost:8102/featured"
    echo "  - All tools: http://localhost:8102/tools"
    echo "  - Health: http://localhost:8102/health"
    echo ""
    exit 0
fi

# Option 2: Start REST API only
if [ "$1" = "rest-only" ]; then
    print_header "Unified REST API"
    start_gateway "Unified-API" "uvicorn qulab.api.main:app --host 0.0.0.0 --port 8000" 8000
    echo ""
    echo "${GREEN}Unified REST API running on http://localhost:8000${NC}"
    echo "  - Docs: http://localhost:8000/docs"
    echo "  - ReDoc: http://localhost:8000/redoc"
    echo ""
    exit 0
fi

# Option 3: Start medical labs only
if [ "$1" = "medical-only" ]; then
    print_header "Medical Diagnostics (10 labs)"
    echo "Starting 10 medical labs on ports 8001–8010..."
    LAB_HOST=0.0.0.0 LAB_PORT_PREFIX=800 bash scripts/start_medical_labs.sh
    echo ""
    echo "${GREEN}Medical labs running:${NC}"
    for port in {8001..8010}; do
        echo "  - http://localhost:$port/docs"
    done
    echo ""
    exit 0
fi

# Default: Start all three gateways
print_header "Gateway 1: MCP HTTP (Materials & R&D / Agents)"
PY="${PYTHON:-python3}"
start_gateway "MCP-Server" "$PY unified_mcp_server.py" 8102
echo "  ℹ️  Featured tools: GET http://localhost:8102/featured"
echo "  ℹ️  Call tools: POST http://localhost:8102/tools/call"
echo ""

print_header "Gateway 2: Unified REST API (Browser / WebSocket)"
start_gateway "Unified-API" "uvicorn qulab.api.main:app --host 0.0.0.0 --port 8000 --reload" 8000
echo "  ℹ️  Interactive docs: http://localhost:8000/docs"
echo "  ℹ️  ReDoc: http://localhost:8000/redoc"
echo ""

print_header "Gateway 3: Medical Diagnostics (10 Microservices)"
echo "${GREEN}Starting 10 medical labs on ports 8001–8010...${NC}"
sleep 1
LAB_HOST=0.0.0.0 LAB_PORT_PREFIX=800 bash scripts/start_medical_labs.sh > logs/medical-labs.log 2>&1 &
sleep 4
echo "${GREEN}✓ Medical labs started${NC}"
echo "  Alzheimer's (8001), Parkinson's (8002), Autoimmune (8003),"
echo "  Sepsis (8004), Wound Healing (8005), Bone Density (8006),"
echo "  Kidney (8007), Liver (8008), Lung (8009), Pain Mgmt (8010)"
echo ""

print_header "Lab Console GUI (React)"
GUI_DIR="qulab-gui"
if [ -d "$GUI_DIR" ]; then
    export PATH="${HOME}/.nvm/versions/node/v22.19.0/bin:${PATH}"
    if command -v npm >/dev/null 2>&1; then
        if [ ! -d "$GUI_DIR/node_modules" ]; then
            echo "${GREEN}Installing GUI dependencies...${NC}"
            (cd "$GUI_DIR" && npm install) >> logs/gui-install.log 2>&1 || true
        fi
        if [ -d "$GUI_DIR/dist" ] || (cd "$GUI_DIR" && npm run build >> ../logs/gui-build.log 2>&1); then
            echo "${GREEN}Starting Lab Console on http://localhost:5173${NC}"
            (cd "$GUI_DIR" && npm run preview -- --host 0.0.0.0 --port 5173) >> logs/gui.log 2>&1 &
            echo $! > logs/gui.pid
            sleep 2
            if command -v open >/dev/null 2>&1; then
                open "http://localhost:5173" 2>/dev/null || true
            fi
        else
            echo "${YELLOW}GUI build failed — see logs/gui-build.log. MCP/API still running.${NC}"
        fi
    else
        echo "${YELLOW}npm not found — skip GUI. Install Node 20+ or use: cd qulab-gui && npm run dev${NC}"
    fi
else
    echo "${YELLOW}qulab-gui/ not found — skip GUI${NC}"
fi
echo ""

print_header "🎉 All gateways running!"
echo ""
echo "${GREEN}✓ MCP Server${NC}            http://localhost:8102       (agents, tools)"
echo "${GREEN}✓ Unified API${NC}          http://localhost:8000       (REST, WebSocket)"
echo "${GREEN}✓ Medical Labs${NC}         http://localhost:8001-8010  (diagnostics)"
echo "${GREEN}✓ Lab Console GUI${NC}      http://localhost:5173       (Product Hunt UI)"
echo ""
echo "${YELLOW}📚 Documentation:${NC}"
echo "  - Frontend wiring: docs/FIGMA_BACKEND_WIRING.md"
echo "  - Production guide: docs/MATERIALS_RD_PRODUCTION.md"
echo "  - Product Hunt: PRODUCT_HUNT.md"
echo ""
echo "${YELLOW}⏹️  To stop all services:${NC}"
echo "  bash STOP_QULAB.sh"
echo ""
echo "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Trap to clean up on exit
trap "echo 'Stopping services...'; bash STOP_QULAB.sh 2>/dev/null || true" EXIT

# Keep script running
wait

