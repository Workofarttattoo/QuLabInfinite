#!/usr/bin/env bash
# Launch all 10 medical FastAPI labs with configurable host and port prefix.
# Usage:
#   LAB_HOST=127.0.0.1 LAB_PORT_PREFIX=90 scripts/start_medical_labs.sh
# Defaults: LAB_HOST=0.0.0.0, LAB_PORT_PREFIX=800 (ports 8001-8010)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${LAB_HOST:-0.0.0.0}"
PORT_PREFIX="${LAB_PORT_PREFIX:-800}"

# Ensure the codebase is importable regardless of where the script is run from.
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

labs=(
  "alzheimers_early_detection:1"
  "parkinsons_progression_predictor:2"
  "autoimmune_disease_classifier:3"
  "sepsis_early_warning:4"
  "wound_healing_optimizer:5"
  "bone_density_predictor:6"
  "kidney_function_calculator:7"
  "liver_disease_staging:8"
  "lung_function_analyzer:9"
  "pain_management_optimizer:10"
)

pids=()

cleanup() {
  if [[ ${#pids[@]} -gt 0 ]]; then
    echo "Stopping medical labs..."
    for pid in "${pids[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
  fi
}

trap cleanup EXIT

for lab in "${labs[@]}"; do
  IFS=":" read -r module suffix <<<"$lab"
  port="${PORT_PREFIX}${suffix}"

  echo "Starting ${module} on ${HOST}:${port}"
  python -m uvicorn "${module}:app" \
    --host "${HOST}" \
    --port "${port}" \
    --app-dir "${ROOT_DIR}" &

  pids+=("$!")
done

wait
