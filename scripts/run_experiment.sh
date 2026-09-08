#!/usr/bin/env bash
# Launch a training run with archived logs + experiments/runs.jsonl tracking.
#
# Usage:
#   bash scripts/run_experiment.sh <experiment_id> <config.json> <gpu> [extra train.py args...]
#
# Env:
#   TRAIN_SCRIPT=train.py            (default)
#   TRAIN_SCRIPT=train_supervised.py (supervised CE baseline)
#   PYTHON=...                       python binary
#
# Example:
#   bash scripts/run_experiment.sh jepa_only configs/cifar100_single_encoder_jepa.json 1
#   TRAIN_SCRIPT=train_supervised.py bash scripts/run_experiment.sh \
#     supervised configs/cifar10_supervised_resnet18.json 1

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <experiment_id> <config.json> <gpu_index> [train args...]" >&2
  exit 1
fi

EXP_ID="$1"
CONFIG="$2"
GPU="$3"
shift 3

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

TRAIN_SCRIPT="${TRAIN_SCRIPT:-train.py}"
if [[ -x /home/sofoklis/miniconda3/bin/python ]]; then
  PYTHON="${PYTHON:-/home/sofoklis/miniconda3/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

RUN_NAME="$("$PYTHON" -c "import json; print(json.load(open('$CONFIG'))['run_name'])")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR" experiments

LOG_FILE="${LOG_DIR}/${STAMP}_${EXP_ID}_gpu${GPU}.log"
REGISTRY="experiments/runs.jsonl"

"$PYTHON" - <<PY
import json
from datetime import datetime, timezone

entry = {
    "experiment_id": "$EXP_ID",
    "run_name": "$RUN_NAME",
    "config": "$CONFIG",
    "train_script": "$TRAIN_SCRIPT",
    "device": "cuda:$GPU",
    "log_file": "$LOG_FILE",
    "status": "started",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "extra_args": "$*",
}
with open("$REGISTRY", "a") as f:
    f.write(json.dumps(entry) + "\n")
print(f"Registered: $REGISTRY")
PY

echo "Starting $EXP_ID ($RUN_NAME) via $TRAIN_SCRIPT on cuda:$GPU"
echo "Log: $LOG_FILE"

nohup env PYTHONPATH=. "$PYTHON" -u "$TRAIN_SCRIPT" \
  --config "$CONFIG" \
  --device "cuda:${GPU}" \
  --output-dir ./results \
  "$@" \
  > "$LOG_FILE" 2>&1 &

echo "PID=$!"
sleep 8
tail -12 "$LOG_FILE"
