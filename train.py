#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tripletjepa.data import DATASETS
from tripletjepa.train import TrainConfig, run_training


def dataset_choices() -> list[str]:
    return sorted(DATASETS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Triplet-JEPA")
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--dataset", type=str, choices=dataset_choices())
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--triplet-weight", type=float)
    parser.add_argument("--negative-mode", type=str, choices=["instance", "class"])
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--train-subset", type=int, help="Use first N train samples (debug)")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets if missing (use on server; off by default)",
    )
    args = parser.parse_args()

    if args.config:
        cfg = TrainConfig.from_json(args.config)
    else:
        cfg = TrainConfig()

    if args.run_name:
        cfg.run_name = args.run_name
    if args.dataset:
        cfg.dataset = args.dataset
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.triplet_weight is not None:
        cfg.triplet_weight = args.triplet_weight
    if args.negative_mode:
        cfg.negative_mode = args.negative_mode
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.train_subset is not None:
        cfg.train_subset = args.train_subset
    if args.download:
        cfg.download = True

    summary = run_training(cfg)
    print(json.dumps(summary["final"], indent=2))


if __name__ == "__main__":
    main()
