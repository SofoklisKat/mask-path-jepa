#!/usr/bin/env python3
"""Plot k-NN vs linear trade-off and learning curves from training histories.

Usage:
  PYTHONPATH=. python scripts/plot_ablation_curves.py \\
    --runs results/paper_cifar100/latent_infonce_jepa_augment \\
               results/paper_cifar100/latent_infonce_jepa_sigreg \\
    --out paper/figures/ablation_curves.pdf

Expects each run directory to contain checkpoint_last.pt with a history list.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_history(run_dir: Path) -> tuple[str, list[dict]]:
    ckpt_path = run_dir / "checkpoint_last.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    name = ckpt.get("config", {}).get("run_name", run_dir.name)
    return name, list(ckpt.get("history", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Run output directories")
    parser.add_argument("--out", type=Path, default=Path("paper/figures/ablation_curves.pdf"))
    parser.add_argument("--scatter-out", type=Path, default=Path("paper/figures/ablation_scatter.pdf"))
    args = parser.parse_args()

    series = [load_history(Path(p)) for p in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, hist in series:
        eval_epochs = [row["epoch"] for row in hist if "knn" in row]
        eval_knn = [row["knn"] for row in hist if "knn" in row]
        eval_lin = [row.get("linear_probe", row.get("linear")) for row in hist if "knn" in row]
        if eval_epochs:
            axes[0].plot(eval_epochs, eval_knn, marker="o", label=name)
            axes[1].plot(eval_epochs, eval_lin, marker="o", label=name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("k-NN@20")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Linear probe")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    for name, hist in series:
        pts = [
            (row.get("linear_probe", row.get("linear")), row["knn"])
            for row in hist
            if "knn" in row
        ]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax2.scatter(xs, ys, label=name)
        ax2.scatter([xs[-1]], [ys[-1]], marker="*", s=120, zorder=5)
    ax2.set_xlabel("Linear probe")
    ax2.set_ylabel("k-NN@20")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig(args.scatter_out, bbox_inches="tight")
    print(f"wrote {args.scatter_out}")


if __name__ == "__main__":
    main()
