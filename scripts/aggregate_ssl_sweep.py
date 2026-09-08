#!/usr/bin/env python3
"""Aggregate the robustness sweep at the fixed epoch-100 endpoint."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


LABELS = {
    "ema": "EMA JEPA cosine",
    "mask_nce": "JEPA InfoNCE (mask only; no EMA)",
    "cos_aug_nce": "Cosine JEPA + augmentation InfoNCE (no EMA)",
    "mvc": "JEPA + MSE/variance/covariance (no EMA)",
    "vicreg": "JEPA + VICReg (no EMA; 25/25/25)",
    "sigreg": "JEPA + SIGReg (no EMA)",
    "cos_only": "Cosine JEPA only (no EMA)",
    "aug_nce_only": "Augmentation InfoNCE only (no EMA)",
}

SEED42 = {
    ("cifar10", "ema"): "results/jepa_resnet18_cifar10/results.json",
    ("cifar10", "mask_nce"): "results/latent_infonce_jepa_cifar10_r18/results.json",
    ("cifar10", "cos_aug_nce"): "results/jepa_cosine_aug_infonce_cifar10_r18/results.json",
    ("cifar10", "mvc"): "results/jepa_cosine_mse_var_cov_cifar10_r18/results.json",
    ("cifar10", "vicreg"): "results/jepa_cosine_vicreg_cifar10_r18/results.json",
    ("cifar10", "sigreg"): "results/jepa_cosine_sigreg_cifar10_r18/results.json",
    ("cifar100", "ema"): "results/cifar100_ema_jepa_r18_seed42/results.json",
    ("cifar100", "mask_nce"): "results/cifar100_mask_only_infonce_r18_seed42/results.json",
    ("cifar100", "cos_aug_nce"): "results/cifar100_cosine_aug_infonce_r18_seed42/results.json",
}

METHODS = {
    "cifar10": ("ema", "mask_nce", "cos_aug_nce", "mvc", "vicreg", "sigreg", "cos_only", "aug_nce_only"),
    "cifar100": ("ema", "mask_nce", "cos_aug_nce", "cos_only", "aug_nce_only"),
}


def result_path(root: Path, dataset: str, method: str, seed: int) -> Path:
    if seed == 42 and (dataset, method) in SEED42:
        return root / SEED42[(dataset, method)]
    name = f"{dataset}_{method}_r18_seed{seed}"
    return root / "results/robustness_sweep_20260731" / name / "results.json"


def endpoint(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    final = data.get("final", {})
    if final.get("epoch") != 100:
        raise ValueError(f"{path}: expected final epoch 100, got {final.get('epoch')}")
    return {
        "knn": float(final["knn"]),
        "linear_probe": float(final["linear_probe"]),
        "elapsed_sec": float(data["elapsed_sec"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output-dir", default="results/robustness_sweep_20260731/aggregate")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    records, missing = [], []
    for dataset, methods in METHODS.items():
        for method in methods:
            for seed in (42, 7, 123):
                path = result_path(root, dataset, method, seed)
                if not path.is_file():
                    missing.append(str(path.relative_to(root)))
                    continue
                values = endpoint(path)
                records.append({"dataset": dataset, "method": method, "label": LABELS[method], "seed": seed, **values})
    if args.require_complete and missing:
        raise SystemExit("Incomplete sweep; missing:\n" + "\n".join(missing))

    summary = []
    for dataset, methods in METHODS.items():
        for method in methods:
            rows = [r for r in records if r["dataset"] == dataset and r["method"] == method]
            if not rows:
                continue
            item = {"dataset": dataset, "method": method, "label": LABELS[method], "n": len(rows)}
            for metric in ("knn", "linear_probe", "elapsed_sec"):
                vals = [r[metric] for r in rows]
                item[f"{metric}_mean"] = statistics.mean(vals)
                item[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else None
            summary.append(item)

    payload = {"endpoint_epoch": 100, "seeds": [42, 7, 123], "complete": not missing, "missing": missing, "records": records, "summary": summary}
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        fields = list(summary[0]) if summary else ["dataset", "method"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    lines = []
    for row in summary:
        def fmt(metric: str) -> str:
            mean, std = 100 * row[f"{metric}_mean"], row[f"{metric}_std"]
            return f"{mean:.2f}" if std is None else f"{mean:.2f} \\pm {100 * std:.2f}"
        lines.append(f"{row['dataset']} & {row['label']} & {fmt('knn')} & {fmt('linear_probe')} & {row['n']} \\\\")
    (output / "table_rows.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {output.relative_to(root)} ({len(records)} runs; {len(missing)} missing)")
    if missing:
        print("missing:")
        print("\n".join(missing))


if __name__ == "__main__":
    main()
