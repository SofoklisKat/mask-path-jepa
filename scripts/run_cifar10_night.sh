#!/usr/bin/env bash
# Overnight CIFAR-10 Contrastive-JEPA suite (ResNet-18, one encoder, no EMA).
# Runs experiments in pairs: two GPUs at a time (GPU + GPU_B).
#
# Usage:
#   nohup bash scripts/run_cifar10_night.sh > experiments/logs/night_launch.log 2>&1 &
#
# Env:
#   GPU=0      first GPU (default 0)
#   GPU_B=1    second GPU (default 1)
#   PYTHON=    python binary
#   EPOCHS=    optional --epochs override

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
GPU_B="${GPU_B:-1}"
EPOCHS="${EPOCHS:-}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="experiments/logs"
mkdir -p "$LOG_DIR" experiments results

if [[ -x /home/sofoklis/miniconda3/bin/python ]]; then
  PYTHON="${PYTHON:-/home/sofoklis/miniconda3/bin/python}"
else
  PYTHON="${PYTHON:-python3}"
fi

EXTRA=()
if [[ -n "$EPOCHS" ]]; then
  EXTRA+=(--epochs "$EPOCHS")
fi

run_one() {
  local name="$1"
  local config="$2"
  local gpu="$3"
  local log="${LOG_DIR}/${STAMP}_${name}_gpu${gpu}.log"

  if [[ ! -f "$config" ]]; then
    echo "[$(date -u +%H:%M:%SZ)] MISSING config: $config" >&2
    return 1
  fi

  echo "================================================================"
  echo "[$(date -u +%H:%M:%SZ)] START $name  config=$config  device=cuda:$gpu"
  echo "  log=$log"
  echo "================================================================"

  set +e
  PYTHONPATH=. "$PYTHON" -u train.py \
    --config "$config" \
    --device "cuda:${gpu}" \
    --output-dir ./results \
    "${EXTRA[@]}" \
    >"$log" 2>&1
  local rc=$?
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "[$(date -u +%H:%M:%SZ)] DONE  $name (exit 0)"
    # Models land under results/<run_name>/ (full ckpt + encoder-only).
    local run_name
    run_name="$("$PYTHON" -c "import json; print(json.load(open('$config'))['run_name'])")"
    local out_dir="./results/${run_name}"
    echo "  models: ${out_dir}/checkpoint_best.pt  ${out_dir}/encoder_best.pt"
    echo "          ${out_dir}/checkpoint_last.pt  ${out_dir}/encoder_last.pt"
    ls -lh "${out_dir}"/checkpoint_*.pt "${out_dir}"/encoder_*.pt 2>/dev/null || true
  else
    echo "[$(date -u +%H:%M:%SZ)] FAIL  $name (exit $rc) — see $log" >&2
  fi
  return $rc
}

MASTER="${LOG_DIR}/${STAMP}_night_master.log"
exec > >(tee -a "$MASTER") 2>&1

echo "Night suite start $STAMP"
echo "  PYTHON=$PYTHON"
echo "  pair GPUs: cuda:$GPU + cuda:$GPU_B"
echo "  master log: $MASTER"
nvidia-smi -L 2>/dev/null || true

# name:config pairs — processed two at a time
# Hybrids = cosine JEPA (mask→clean) + aug/clean regularizer (no JEPA-InfoNCE).
CONFIGS=(
  "jepa_mse_var_cov:configs/cifar10_jepa_mse_var_cov.json"
  "jepa_sigreg:configs/cifar10_jepa_sigreg.json"
  "jepa_vicreg:configs/cifar10_jepa_vicreg.json"
  "jepa_aug_infonce:configs/cifar10_latent_infonce_jepa.json"
)

FAILED=0
i=0
n=${#CONFIGS[@]}

while (( i < n )); do
  entry_a="${CONFIGS[$i]}"
  name_a="${entry_a%%:*}"
  config_a="${entry_a#*:}"

  if (( i + 1 < n )); then
    entry_b="${CONFIGS[$((i + 1))]}"
    name_b="${entry_b%%:*}"
    config_b="${entry_b#*:}"

    echo "---- pair: $name_a @ cuda:$GPU  ||  $name_b @ cuda:$GPU_B ----"
    run_one "$name_a" "$config_a" "$GPU" &
    PID_A=$!
    run_one "$name_b" "$config_b" "$GPU_B" &
    PID_B=$!
    wait "$PID_A" || FAILED=$((FAILED + 1))
    wait "$PID_B" || FAILED=$((FAILED + 1))
    i=$((i + 2))
  else
    echo "---- solo: $name_a @ cuda:$GPU ----"
    run_one "$name_a" "$config_a" "$GPU" || FAILED=$((FAILED + 1))
    i=$((i + 1))
  fi
done

echo "================================================================"
echo "[$(date -u +%H:%M:%SZ)] Night suite finished. failures=$FAILED"
echo "Logs under $LOG_DIR/${STAMP}_*"
echo "================================================================"
exit "$FAILED"
