#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs results

CKPT=""
for candidate in \
  "results/latent_infonce_jepa_sigreg/checkpoint_last.pt" \
  "results/paper_cifar100/latent_infonce_jepa_sigreg/checkpoint_last.pt"; do
  if [[ -f "$candidate" ]]; then
    CKPT="$candidate"
    break
  fi
done

if [[ -z "$CKPT" ]]; then
  echo "No checkpoint found. Expected one of:" >&2
  echo "  results/latent_infonce_jepa_sigreg/checkpoint_last.pt" >&2
  echo "  results/paper_cifar100/latent_infonce_jepa_sigreg/checkpoint_last.pt" >&2
  exit 1
fi

echo "Resuming from: $CKPT"
echo "Target: epochs 500 on cuda:1"

exec env PYTHONPATH=. python -u train.py \
  --config configs/cifar100_latent_infonce_jepa_sigreg.json \
  --epochs 500 \
  --resume "$CKPT" \
  --device cuda:1
