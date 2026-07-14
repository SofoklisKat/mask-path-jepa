#!/usr/bin/env python3
"""Run the full Triplet-JEPA experiment suite and aggregate results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tripletjepa.data import DATASETS
from tripletjepa.train import TrainConfig, run_training

# Shared hyperparameters merged into every preset (dataset-specific overrides below).
BASE_DEFAULTS: dict = {
    "epochs": 100,
    "eval_every": 10,
    "embed_dim": 256,
    "ema_momentum": 0.996,
    "lr": 3e-4,
    "margin": 0.2,
    "mask_ratio": 0.6,
    "knn_k": 20,
    "probe_epochs": 50,
    "seed": 42,
}

PRESETS: dict[str, dict] = {
    # --- EMA + predictor (original protocol) ---
    "jepa_baseline": {
        "run_name": "jepa_baseline",
        "training_mode": "jepa_ema",
        "triplet_weight": 0.0,
        "negative_mode": "instance",
        "corrupt_schedule": "block",
    },
    "jepa_blur_curriculum": {
        "run_name": "jepa_blur_curriculum",
        "training_mode": "jepa_ema",
        "triplet_weight": 0.0,
        "negative_mode": "instance",
        "corrupt_schedule": "blur_to_mask",
        "blur_sigma_min": 0.5,
        "blur_sigma_max": 3.0,
        "mask_ratio": 1.0,
    },
    "triplet_instance": {
        "run_name": "triplet_instance",
        "training_mode": "jepa_ema",
        "triplet_weight": 0.5,
        "negative_mode": "instance",
    },
    "triplet_class": {
        "run_name": "triplet_class",
        "training_mode": "jepa_ema",
        "triplet_weight": 0.5,
        "negative_mode": "class",
    },
    "triplet_scramble": {
        "run_name": "triplet_scramble",
        "training_mode": "jepa_ema",
        "triplet_weight": 0.05,
        "negative_mode": "scramble",
        "scramble_patch_size": 4,
    },
    # --- Single encoder + triplet regularizer ---
    "latent_triplet_jepa": {
        "run_name": "latent_triplet_jepa",
        "training_mode": "latent_triplet",
        "anchor_mode": "encoder_corrupt",
        "triplet_weight": 0.0,
        "negative_mode": "scramble",
        "scramble_patch_size": 4,
    },
    "latent_triplet": {
        "run_name": "latent_triplet",
        "training_mode": "latent_triplet",
        "anchor_mode": "encoder_corrupt",
        "triplet_weight": 0.05,
        "negative_mode": "scramble_class",
        "scramble_patch_size": 4,
    },
    "latent_triplet_heavy": {
        "run_name": "latent_triplet_heavy",
        "training_mode": "latent_triplet",
        "anchor_mode": "encoder_corrupt",
        "triplet_weight": 25.0,
        "negative_mode": "scramble_class",
        "scramble_patch_size": 4,
    },
    "latent_triplet_align_heavy": {
        "run_name": "latent_triplet_align_heavy",
        "training_mode": "latent_triplet",
        "anchor_mode": "predictor_corrupt",
        "triplet_weight": 25.0,
        "negative_mode": "scramble_class",
        "scramble_patch_size": 4,
    },
    "triplet_heavy_ema": {
        "run_name": "triplet_heavy_ema",
        "training_mode": "jepa_ema",
        "triplet_weight": 25.0,
        "negative_mode": "scramble_class",
        "scramble_patch_size": 4,
    },
    "single_encoder_jepa": {
        "run_name": "single_encoder_jepa",
        "training_mode": "latent_triplet",
        "anchor_mode": "encoder_corrupt",
        "use_ema_target": False,
        "triplet_weight": 0.0,
        "negative_mode": "instance",
    },
    "single_encoder_scramble": {
        "run_name": "single_encoder_scramble",
        "training_mode": "latent_triplet",
        "anchor_mode": "encoder_corrupt",
        "use_ema_target": False,
        "triplet_weight": 0.05,
        "negative_mode": "scramble",
        "scramble_patch_size": 4,
    },
    # --- Single encoder + VICReg (no EMA) ---
    "latent_vicreg_encoder": {
        "run_name": "latent_vicreg_encoder",
        "training_mode": "latent_vicreg",
        "anchor_mode": "encoder_corrupt",
        "triplet_weight": 0.0,
        "vicreg_var_weight": 25.0,
        "vicreg_cov_weight": 25.0,
        "corrupt_schedule": "block",
    },
    "latent_vicreg_align": {
        "run_name": "latent_vicreg_align",
        "training_mode": "latent_vicreg",
        "anchor_mode": "predictor_corrupt",
        "triplet_weight": 0.0,
        "vicreg_var_weight": 25.0,
        "vicreg_cov_weight": 25.0,
        "corrupt_schedule": "block",
    },
    "latent_vicreg_encoder_curriculum": {
        "run_name": "latent_vicreg_encoder_curriculum",
        "training_mode": "latent_vicreg",
        "anchor_mode": "encoder_corrupt",
        "triplet_weight": 0.0,
        "vicreg_var_weight": 25.0,
        "vicreg_cov_weight": 25.0,
        "corrupt_schedule": "mask_curriculum",
        "mask_ratio": 0.6,
    },
    "latent_vicreg_align_curriculum": {
        "run_name": "latent_vicreg_align_curriculum",
        "training_mode": "latent_vicreg",
        "anchor_mode": "predictor_corrupt",
        "triplet_weight": 0.0,
        "vicreg_var_weight": 25.0,
        "vicreg_cov_weight": 25.0,
        "corrupt_schedule": "mask_curriculum",
        "mask_ratio": 0.6,
    },
}

SUITES: dict[str, list[str]] = {
    "core": [
        "jepa_baseline",
        "triplet_instance",
        "triplet_class",
    ],
    "ema": [
        "jepa_baseline",
        "triplet_instance",
        "triplet_class",
        "triplet_scramble",
    ],
    "latent_triplet": [
        "latent_triplet_jepa",
        "latent_triplet",
        "latent_triplet_heavy",
        "latent_triplet_align_heavy",
        "single_encoder_jepa",
        "single_encoder_scramble",
    ],
    "ablation": [
        "latent_triplet_heavy",
        "latent_triplet_align_heavy",
        "triplet_heavy_ema",
    ],
    "latent_vicreg": [
        "latent_vicreg_encoder",
        "latent_vicreg_align",
        "latent_vicreg_encoder_curriculum",
        "latent_vicreg_align_curriculum",
    ],
    "full": [
        "jepa_baseline",
        "jepa_blur_curriculum",
        "triplet_instance",
        "triplet_class",
        "triplet_scramble",
        "latent_triplet_jepa",
        "latent_triplet",
        "latent_vicreg_encoder",
        "latent_vicreg_align",
        "latent_vicreg_encoder_curriculum",
        "latent_vicreg_align_curriculum",
    ],
}


def dataset_defaults(dataset: str) -> dict:
    batch_size = 256 if dataset in {"cifar100", "tiny_imagenet"} else 128
    return {"batch_size": batch_size}


def aggregate(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*/results.json")):
        data = json.loads(path.read_text())
        final = data["final"]
        rows.append(
            {
                "run": data["run_name"],
                "training_mode": data.get("training_mode", "jepa_ema"),
                "use_ema_target": data.get("use_ema_target", True),
                "anchor_mode": data.get("anchor_mode", "predictor_corrupt"),
                "triplet_weight": data.get("triplet_weight", 0.0),
                "negative_mode": data.get("negative_mode", "instance"),
                "vicreg_var_weight": data.get("vicreg_var_weight", 0.0),
                "vicreg_cov_weight": data.get("vicreg_cov_weight", 0.0),
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
    header = (
        "| Method | Mode | EMA | Anchor | λ | Negatives | VICReg | "
        "k-NN | Linear | Time (s) |\n"
    )
    header += (
        "|--------|------|-----|--------|---|-----------|--------|"
        "------|--------|----------|\n"
    )
    lines = []
    for r in rows:
        lam = "0" if r["triplet_weight"] == 0 else str(r["triplet_weight"])
        ema = "yes" if r.get("use_ema_target", True) else "no"
        vicreg = (
            f"{r['vicreg_var_weight']}/{r['vicreg_cov_weight']}"
            if r["vicreg_var_weight"] or r["vicreg_cov_weight"]
            else "—"
        )
        knn = f"{r['knn']:.3f}" if r["knn"] is not None else "—"
        linear = f"{r['linear_probe']:.3f}" if r["linear_probe"] is not None else "—"
        elapsed = r["elapsed_sec"] if r["elapsed_sec"] is not None else "—"
        lines.append(
            f"| {r['run']} | {r['training_mode']} | {ema} | {r['anchor_mode']} | "
            f"{lam} | {r['negative_mode']} | {vicreg} | {knn} | {linear} | {elapsed} |"
        )
    return header + "\n".join(lines)


def is_complete(results_dir: Path, run_name: str) -> bool:
    path = results_dir / run_name / "results.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    return "final" in data and data["final"].get("knn") is not None


def resolve_suite(name: str, only: list[str] | None) -> list[str]:
    if only:
        unknown = [x for x in only if x not in PRESETS]
        if unknown:
            raise ValueError(f"Unknown experiment(s): {unknown}. Choose from: {sorted(PRESETS)}")
        return only
    if name not in SUITES:
        raise ValueError(f"Unknown suite={name!r}. Choose from: {sorted(SUITES)}")
    return SUITES[name]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Triplet-JEPA experiment suites and aggregate results."
    )
    parser.add_argument("--dataset", default="cifar100", choices=sorted(DATASETS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--suite",
        default="full",
        choices=sorted(SUITES),
        help="Preset group to run (default: full)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="EXPERIMENT",
        help="Run specific experiments from PRESETS (overrides --suite)",
    )
    parser.add_argument("--quick", action="store_true", help="20 epochs, 5k train subset")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have results.json with k-NN",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets if missing (off by default)",
    )
    args = parser.parse_args()

    out = Path(args.output_dir or f"./results/paper_{args.dataset}")
    experiments = resolve_suite(args.suite, args.only)

    if not args.aggregate_only:
        epochs = 20 if args.quick else (args.epochs or BASE_DEFAULTS["epochs"])
        base = {
            **BASE_DEFAULTS,
            **dataset_defaults(args.dataset),
            "dataset": args.dataset,
            "epochs": epochs,
            "eval_every": 5 if args.quick else BASE_DEFAULTS["eval_every"],
            "output_dir": str(out),
            "train_subset": 5000 if args.quick else None,
            "download": args.download,
        }

        print(f"Suite: {args.suite if not args.only else 'custom'}")
        print(f"Dataset: {args.dataset} | Epochs: {epochs} | Output: {out}")
        print(f"Experiments ({len(experiments)}): {', '.join(experiments)}\n")

        for preset_name in experiments:
            preset = PRESETS[preset_name]
            run_name = preset["run_name"]
            if args.skip_existing and is_complete(out, run_name):
                print(f"=== Skipping {preset_name} ({run_name}) — already complete ===")
                continue
            cfg_dict = {**base, **preset}
            print(f"\n=== Running {preset_name} ({run_name}) ===")
            run_training(TrainConfig(**cfg_dict))

    rows = aggregate(out)
    table = to_markdown_table(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "TABLE.md").write_text(table + "\n")
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\n" + table)
    print(f"\nWrote {out / 'TABLE.md'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
