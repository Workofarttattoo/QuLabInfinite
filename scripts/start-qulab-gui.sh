#!/usr/bin/env bash
# Start QuLab Infinite GUI (avoids port 3000 = Grafana in docker-compose)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GUI="$ROOT/qulab-gui"

if [ -f "$HOME/.nvm/nvm.sh" ]; then
  # shellcheck source=/dev/null
  . "$HOME/.nvm/nvm.sh"
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Node/npm not found. Install Node 20+ or run: nvm install 22"
  exit 1
fi

cd "$GUI"
if [ ! -d node_modules ]; then
  echo "Installing dependencies..."
  npm install
fi

echo ""
echo "QuLab GUI → http://127.0.0.1:5173/labs/materials"
echo "Stop with Ctrl+C"
echo ""

npm run dev -- --host 127.0.0.1 --port 5173
