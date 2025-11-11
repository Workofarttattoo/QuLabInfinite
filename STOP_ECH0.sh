#!/bin/bash
# STOP ECH0 - Graceful shutdown
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

echo "🛑 Stopping ECH0 14B..."

if [ -f ech0.pid ]; then
    PIDS=$(cat ech0.pid)
    for PID in $PIDS; do
        if ps -p $PID > /dev/null 2>&1; then
            echo "   Stopping process $PID"
            kill $PID
        fi
    done
    rm -f ech0.pid
    echo "✅ ECH0 stopped"
else
    echo "⚠️  No PID file found, killing by name..."
    pkill -f "ech0_mcp_server" 2>/dev/null || true
    pkill -f "ech0_lab_director" 2>/dev/null || true
    echo "✅ Cleanup complete"
fi

echo ""
echo "ECH0 is now offline."
echo "To restart: ./ACTIVATE_ECH0.sh"
echo ""
