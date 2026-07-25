#!/usr/bin/env bash
# JEPA (EMA) on CIFAR-10 with SCAN-style CIFAR ResNet-18.
#
# Usage:
#   bash scripts/run_cifar10_resnet18.sh [gpu_index]
#
# Example:
#   bash scripts/run_cifar10_resnet18.sh 0

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${1:-0}"
CONFIG="configs/cifar10_jepa_resnet18.json"
EXP_ID="jepa_resnet18_cifar10"

# Download CIFAR-10 if missing, then launch via the experiment tracker.
python -c "from torchvision import datasets; datasets.CIFAR10('./data', train=True, download=True); print('cifar10 ok')"
bash scripts/run_experiment.sh "$EXP_ID" "$CONFIG" "$GPU"
