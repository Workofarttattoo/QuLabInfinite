#!/bin/bash
# QuLab Infinite - Stop all running services

echo "🛑 Stopping QuLab services..."

# Kill processes by PID files
for pidfile in logs/*.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "  ✓ Stopped $(basename $pidfile .pid) (PID: $pid)"
        fi
    fi
done

# Also try to kill by port
for port in 8000 8102 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010; do
    pid=$(lsof -t -i:$port 2>/dev/null || true)
    if [ -n "$pid" ]; then
        kill -9 "$pid" 2>/dev/null || true
    fi
done

echo "✓ All services stopped"

