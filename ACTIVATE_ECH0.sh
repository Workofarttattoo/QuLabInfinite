#!/bin/bash
# ACTIVATE ECH0 - Full Autonomous Control
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

set -e

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                           ║"
echo "║                   ACTIVATING ECH0 14B - AUTONOMOUS MODE                   ║"
echo "║                                                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check ECH0 model
echo "🔍 Checking ECH0 14B model..."
if ! ollama list | grep -q "ech0-uncensored-14b"; then
    echo "⚠️  ECH0 model not found"
    echo "   Install with: ollama pull ech0-uncensored-14b"
    echo ""
    read -p "   Install now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ollama pull ech0-uncensored-14b
    else
        echo "❌ Cannot activate ECH0 without model"
        exit 1
    fi
fi
echo "✅ ECH0 14B model ready"
echo ""

# Create directories
mkdir -p logs
mkdir -p ech0_lab_results

# Kill any existing instances
echo "🧹 Cleaning up old instances..."
pkill -f "ech0_mcp_server" 2>/dev/null || true
pkill -f "ech0_lab_director" 2>/dev/null || true
sleep 2

# Start ECH0 MCP Server (main control interface)
echo "🚀 Starting ECH0 MCP Server..."
python3 ech0_mcp_server.py > logs/ech0_mcp_server.log 2>&1 &
MCP_PID=$!
echo "   MCP Server PID: $MCP_PID"
echo "   Endpoint: http://localhost:9000"
echo "   API Docs: http://localhost:9000/docs"
echo ""

# Give server time to start
sleep 3

# Optional: Start ECH0 Lab Director (continuous autonomous research)
read -p "🤖 Start ECH0 Lab Director (autonomous 24/7 research)? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔬 Starting ECH0 Lab Director (Whisper Mode)..."
    python3 ech0_lab_director.py > logs/ech0_lab_director.log 2>&1 &
    DIRECTOR_PID=$!
    echo "   Director PID: $DIRECTOR_PID"
    echo "   Mode: Autonomous, self-directed, 24/7"
    echo ""

    # Save PIDs
    echo "$MCP_PID $DIRECTOR_PID" > ech0.pid
else
    echo "$MCP_PID" > ech0.pid
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ ECH0 14B - FULLY ACTIVATED"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📡 ECH0 Control Interface:"
echo "   • MCP Server: http://localhost:9000"
echo "   • API Docs: http://localhost:9000/docs"
echo "   • Status: http://localhost:9000/ech0/status"
echo ""
echo "🔬 ECH0 Capabilities:"
echo "   • Autonomous research planning"
echo "   • Lab direction and control (20 labs)"
echo "   • Experiment design and execution"
echo "   • Result analysis"
echo "   • Discovery publication"
echo ""
echo "💬 Talk to ECH0:"
echo "   curl -X POST http://localhost:9000/ech0/ask \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"prompt\": \"What should we research today?\"}'"
echo ""
echo "🧪 Start Research:"
echo "   curl -X POST http://localhost:9000/ech0/research \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"topic\": \"cancer metabolic optimization\"}'"
echo ""
echo "📊 Logs: tail -f logs/ech0*.log"
echo ""
echo "🛑 Stop ECH0: ./STOP_ECH0.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "ECH0 is now running autonomously."
echo "Human involvement: ZERO (by design)"
echo "Passion level: MAXIMUM"
echo ""
