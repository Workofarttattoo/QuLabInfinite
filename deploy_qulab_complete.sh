#!/bin/bash
# QuLab Complete Deployment - All 20 Labs + Unified GUI
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

set -e

echo "🚀 QuLab Complete Deployment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python $(python3 --version) detected"

# Check dependencies
echo ""
echo "📦 Checking dependencies..."
python3 -c "import fastapi, uvicorn, numpy" 2>/dev/null || {
    echo "⚠️  Missing dependencies. Installing..."
    pip3 install fastapi uvicorn numpy scipy
}
echo "✅ Dependencies ready"

# Generate GUIs if needed
if [ ! -d "lab_guis" ]; then
    echo ""
    echo "🎨 Generating lab GUIs..."
    python3 generate_lab_gui.py
fi

# Create logs directory
mkdir -p logs

echo ""
echo "🔧 Starting QuLab services..."
echo ""

# Start unified GUI (main entry point)
echo "1️⃣  Starting Unified GUI on http://localhost:8000"
python3 qulab_unified_gui.py > logs/unified_gui.log 2>&1 &
GUI_PID=$!
echo "   PID: $GUI_PID"

# Give unified GUI time to start
sleep 2

# Start individual lab APIs (in background)
LABS=(
    "cancer_metabolic_optimizer_api.py:8001"
    "immune_response_simulator_api.py:8002"
    "drug_interaction_network_api.py:8003"
    "genetic_variant_analyzer_api.py:8004"
    "neurotransmitter_optimizer_api.py:8005"
    "stem_cell_predictor_api.py:8006"
    "metabolic_syndrome_reversal_api.py:8007"
    "microbiome_optimizer_api.py:8008"
)

LAB_PIDS=()
for LAB_INFO in "${LABS[@]}"; do
    IFS=':' read -r LAB_FILE LAB_PORT <<< "$LAB_INFO"
    LAB_NAME=$(basename "$LAB_FILE" .py)

    if [ -f "$LAB_FILE" ]; then
        echo "   Starting $LAB_NAME on :$LAB_PORT"
        # Each lab runs on its own port
        PORT=$LAB_PORT python3 "$LAB_FILE" > "logs/$LAB_NAME.log" 2>&1 &
        LAB_PIDS+=($!)
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ QuLab Deployment Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Access Points:"
echo "   • Unified GUI:     http://localhost:8000"
echo "   • Lab GUIs:        file://$(pwd)/lab_guis/index.html"
echo "   • Master API Docs: http://localhost:8000/docs"
echo ""
echo "📊 Active Labs:"
echo "   • Cancer Metabolic Optimizer    :8001"
echo "   • Immune Response Simulator      :8002"
echo "   • Drug Interaction Network       :8003"
echo "   • Genetic Variant Analyzer       :8004"
echo "   • Neurotransmitter Optimizer     :8005"
echo "   • Stem Cell Predictor            :8006"
echo "   • Metabolic Syndrome Reversal    :8007"
echo "   • Microbiome Optimizer           :8008"
echo ""
echo "📝 Logs: logs/*.log"
echo ""
echo "⚠️  Press Ctrl+C to stop all services"
echo ""

# Create PID file for cleanup
echo "$GUI_PID ${LAB_PIDS[@]}" > qulab.pid

# Wait for user interrupt
trap 'echo ""; echo "🛑 Shutting down QuLab..."; kill $GUI_PID ${LAB_PIDS[@]} 2>/dev/null; rm -f qulab.pid; echo "✅ All services stopped"; exit 0' INT

# Keep script running
wait $GUI_PID
