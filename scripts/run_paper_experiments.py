#!/usr/bin/env python3
"""Run the low-resource paper experiment suite and aggregate results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tripletjepa.data import DATASETS
from tripletjepa.train import TrainConfig, run_training


PRESETS: dict[str, dict] = {
    "jepa_baseline": {
        "run_name": "jepa_baseline",
        "triplet_weight": 0.0,
        "negative_mode": "instance",
    },
    "triplet_instance": {
        "run_name": "triplet_instance",
        "triplet_weight": 0.5,
        "negative_mode": "instance",
    },
    "triplet_class": {
        "run_name": "triplet_class",
        "triplet_weight": 0.5,
        "negative_mode": "class",
    },
    "triplet_scramble": {
        "run_name": "triplet_scramble",
        "triplet_weight": 0.5,
        "negative_mode": "scramble",
        "scramble_patch_size": 4,
    },
    "latent_triplet_jepa": {
        "run_name": "latent_triplet_jepa",
        "training_mode": "latent_triplet",
        "triplet_weight": 0.0,
        "negative_mode": "scramble",
        "scramble_patch_size": 4,
    },
    "latent_triplet": {
        "run_name": "latent_triplet",
        "training_mode": "latent_triplet",
        "triplet_weight": 0.05,
        "negative_mode": "scramble_class",
        "scramble_patch_size": 4,
    },
}


def aggregate(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*/results.json")):
        data = json.loads(path.read_text())
        final = data["final"]
        rows.append(
            {
                "run": data["run_name"],
                "triplet_weight": data["triplet_weight"],
                "negative_mode": data["negative_mode"],
                "knn": final.get("knn"),
                "linear_probe": final.get("linear_probe"),
                "loss": final.get("loss"),
                "elapsed_sec": data.get("elapsed_sec"),
            }
        )
    return rows


def to_markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_No results yet._"
    header = "| Method | λ | Negatives | k-NN | Linear probe | Time (s) |\n"
    header += "|--------|---|-----------|------|--------------|----------|\n"
    lines = []
    for r in rows:
        lam = "0" if r["triplet_weight"] == 0 else str(r["triplet_weight"])
        lines.append(
            f"| {r['run']} | {lam} | {r['negative_mode']} | "
            f"{r['knn']:.3f} | {r['linear_probe']:.3f} | {r['elapsed_sec']} |"
        )
    return header + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", default="./results/paper_cifar10")
    parser.add_argument("--quick", action="store_true", help="20 epochs, 5k train subset")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--with-scramble",
        action="store_true",
        help="Also run triplet_scramble preset (EMA + predictor)",
    )
    parser.add_argument(
        "--with-latent-triplet",
        action="store_true",
        help="Also run latent_triplet_jepa and latent_triplet (single encoder, no EMA)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets if missing (off by default)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    if not args.aggregate_only:
        base = {
            "dataset": args.dataset,
            "epochs": 20 if args.quick else args.epochs,
            "eval_every": 5 if args.quick else 10,
            "output_dir": str(out),
            "train_subset": 5000 if args.quick else None,
            "download": args.download,
        }
        presets = dict(PRESETS)
        if not args.with_scramble:
            presets.pop("triplet_scramble", None)
        if not args.with_latent_triplet:
            presets.pop("latent_triplet_jepa", None)
            presets.pop("latent_triplet", None)
        for preset_name, preset in presets.items():
            cfg_dict = {**base, **preset}
            print(f"\n=== Running {preset_name} ===")
            run_training(TrainConfig(**cfg_dict))

    rows = aggregate(out)
    table = to_markdown_table(rows)
    (out / "TABLE.md").write_text(table + "\n")
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\n" + table)


if __name__ == "__main__":
    main()
