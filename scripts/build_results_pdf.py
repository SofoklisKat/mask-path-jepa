#!/usr/bin/env python3
"""Build paper/results.pdf from checkpoint histories and ablation figures."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
FIG = PAPER / "figures"


def load_run(run_dir: Path) -> dict:
    ckpt = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    hist = ckpt.get("history", [])
    eval_rows = [r for r in hist if "knn" in r]
    best_knn = max(eval_rows, key=lambda r: r["knn"]) if eval_rows else {}
    best_lin = max(eval_rows, key=lambda r: r.get("linear_probe", 0)) if eval_rows else {}
    last = eval_rows[-1] if eval_rows else {}
    return {
        "name": cfg.get("run_name", run_dir.name),
        "mode": cfg.get("training_mode", "?"),
        "backbone": cfg.get("backbone", "resnet"),
        "epoch": ckpt.get("epoch", "?"),
        "best_knn": best_knn.get("knn"),
        "best_knn_ep": best_knn.get("epoch"),
        "best_lin": best_lin.get("linear_probe"),
        "best_lin_ep": best_lin.get("epoch"),
        "last_knn": last.get("knn"),
        "last_lin": last.get("linear_probe"),
        "last_ep": last.get("epoch"),
        "history": hist,
    }


def text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.94, title, fontsize=16, fontweight="bold", va="top")
    y = 0.88
    for para in body.strip().split("\n\n"):
        fig.text(0.07, y, textwrap.fill(para, width=95), fontsize=10, va="top", family="monospace")
        y -= 0.06 * (1 + para.count("\n") + len(para) // 90)
    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def table_page(pdf: PdfPages, rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    cols = ["Run", "Backbone", "Ep", "Best k-NN", "@", "Best Linear", "@", "Last k-NN", "Last Linear"]
    data = []
    for r in rows:
        data.append([
            r["name"][:28],
            r["backbone"],
            str(r["epoch"]),
            f"{r['best_knn']:.3f}" if r["best_knn"] is not None else "---",
            str(r["best_knn_ep"] or ""),
            f"{r['best_lin']:.3f}" if r["best_lin"] is not None else "---",
            str(r["best_lin_ep"] or ""),
            f"{r['last_knn']:.3f}" if r["last_knn"] is not None else "---",
            f"{r['last_lin']:.3f}" if r["last_lin"] is not None else "---",
        ])
    table = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("CIFAR-100 SSL Results (frozen encoder eval)", fontsize=14, pad=20)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def embed_pdf_page(pdf: PdfPages, image_path: Path, title: str) -> None:
    if not image_path.is_file():
        return
    if image_path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            from matplotlib import image as mpimg
            import io
            # Render via matplotlib from PDF is awkward; skip if no png
        except ImportError:
            pass
    # Prefer PNG; for PDF figures convert name
    png = image_path.with_suffix(".png")
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    if png.is_file():
        ax.imshow(plt.imread(png))
    else:
        # Re-generate curves inline if only pdf exists - read via subprocess convert?
        ax.text(0.5, 0.5, f"See {image_path.name}", ha="center", va="center")
    ax.axis("off")
    ax.set_title(title, fontsize=13)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_curves_page(pdf: PdfPages, runs: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    for run in runs:
        pts = [(r["epoch"], r["knn"], r.get("linear_probe")) for r in run["history"] if "knn" in r]
        if not pts:
            continue
        eps, knns, lins = zip(*pts)
        axes[0].plot(eps, knns, marker="o", label=run["name"])
        axes[1].plot(eps, lins, marker="o", label=run["name"])
    axes[0].set(xlabel="Epoch", ylabel="k-NN@20", title="k-NN learning curves")
    axes[1].set(xlabel="Epoch", ylabel="Linear probe", title="Linear probe learning curves")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Training curves from checkpoints", fontsize=14)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_page(pdf: PdfPages, runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 8.27))
    for run in runs:
        pts = [(r.get("linear_probe"), r["knn"], r["epoch"]) for r in run["history"] if "knn" in r]
        if not pts:
            continue
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, alpha=0.5, label=run["name"])
        ax.scatter([xs[-1]], [ys[-1]], s=80, marker="*", zorder=5)
    ax.set(xlabel="Linear probe", ylabel="k-NN@20", title="k-NN vs Linear trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    run_dirs = sorted({p.parent for p in RESULTS.rglob("checkpoint_last.pt")})
    runs = [load_run(d) for d in run_dirs if (d / "checkpoint_last.pt").is_file()]

    out = PAPER / "results.pdf"
    PAPER.mkdir(parents=True, exist_ok=True)

    protocol = """
Evaluation protocol (CIFAR-100, frozen encoder):
  - k-NN@20 on L2-normalized features (cosine distance)
  - Linear probe: SGD 50 epochs on train features, test accuracy
  - Corrupt view: 60% block mask (4x4 patches)
  - SmallResNet ~2.8M params unless backbone=resnet50

Completed runs loaded from results/*/checkpoint_last.pt
"""

    with PdfPages(out) as pdf:
        text_page(
            pdf,
            "JEPA — Results Summary",
            protocol + "\n\nGenerated from checkpoint histories.\nLaTeX source: paper/main.tex",
        )
        table_page(pdf, runs)
        if runs:
            plot_curves_page(pdf, runs)
            plot_scatter_page(pdf, runs)

    print(f"wrote {out}")

    # Also export PNGs for LaTeX includegraphics fallback
    if runs:
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        for run in runs:
            pts = [(r["epoch"], r["knn"], r.get("linear_probe")) for r in run["history"] if "knn" in r]
            if not pts:
                continue
            eps, knns, lins = zip(*pts)
            axes[0].plot(eps, knns, marker="o", label=run["name"])
            axes[1].plot(eps, lins, marker="o", label=run["name"])
        for ax, ylab in zip(axes, ["k-NN@20", "Linear probe"]):
            ax.set(xlabel="Epoch", ylabel=ylab)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        png_curves = FIG / "ablation_curves.png"
        fig.savefig(png_curves, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 8))
        for run in runs:
            pts = [(r.get("linear_probe"), r["knn"]) for r in run["history"] if "knn" in r]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, alpha=0.5, label=run["name"])
            ax.scatter([xs[-1]], [ys[-1]], s=80, marker="*")
        ax.set(xlabel="Linear probe", ylabel="k-NN@20")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        png_scatter = FIG / "ablation_scatter.png"
        fig.savefig(png_scatter, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png_curves}")
        print(f"wrote {png_scatter}")


if __name__ == "__main__":
    main()
