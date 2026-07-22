#!/usr/bin/env python3
"""Toy figure: what good k-NN from InfoNCE means in embedding space.

  PYTHONPATH=. python scripts/visualize_knn_geometry_toy.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np


def main() -> None:
    out = Path("results/analysis/knn_geometry_toy.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # --- Left: InfoNCE-like layout (good k-NN) ---
    ax = axes[0]
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls="--", color="gray", lw=0.8, alpha=0.5))

    # 3 classes, but instance clusters overlap in 2D projection
    # Each "star" is one training image; small dots are aug/mask views of same image
    centers = {
        "cat instances": ([-0.55, 0.35], [-0.35, 0.55], [-0.15, 0.25]),
        "dog instances": ([0.45, 0.45], [0.65, 0.25], [0.25, 0.15]),
        "truck instances": ([-0.2, -0.55], [0.15, -0.65], [0.4, -0.45]),
    }
    colors = {"cat instances": "#e74c3c", "dog instances": "#3498db", "truck instances": "#2ecc71"}
    rng = np.random.default_rng(0)

    for label, pts in centers.items():
        c = colors[label]
        for i, center in enumerate(pts):
            cx, cy = center
            ax.scatter([cx], [cy], c=c, s=140, edgecolors="black", linewidths=0.7, zorder=3)
            ax.annotate(f"{label[0].upper()}{i+1}", (cx, cy), xytext=(5, 5), textcoords="offset points", fontsize=8)
            # views of same image (aug / mask) stay close
            for _ in range(3):
                dx, dy = rng.normal(0, 0.04, 2)
                ax.scatter([cx + dx], [cy + dy], c=c, s=25, alpha=0.6, zorder=2)

    # test query (cat) — neighbors are mostly cats even though classes overlap
    qx, qy = -0.42, 0.42
    ax.scatter([qx], [qy], c="black", s=200, marker="*", zorder=5, label="test image")
    neighbors = [(-0.55, 0.35), (-0.35, 0.55), (-0.15, 0.25)]
    for nx, ny in neighbors:
        ax.plot([qx, nx], [qy, ny], "k-", alpha=0.35, lw=1)
        ax.scatter([nx], [ny], facecolors="none", edgecolors="black", s=180, linewidths=1.5, zorder=4)
    ax.annotate("k-NN vote → cat", (qx, qy), xytext=(-1.05, 0.85), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("InfoNCE-like geometry (good k-NN)\nEach image = tight cloud; neighbors share label locally")
    ax.axis("off")

    # --- Right: same points but linear boundary fails ---
    ax = axes[1]
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls="--", color="gray", lw=0.8, alpha=0.5))

    for label, pts in centers.items():
        c = colors[label]
        for i, center in enumerate(pts):
            cx, cy = center
            ax.scatter([cx], [cy], c=c, s=140, edgecolors="black", linewidths=0.7, zorder=3)

    # Attempted linear class boundaries (lines in 2D) — cannot separate well
    ax.plot([-1.1, 1.1], [0.15, 0.55], "purple", ls="--", lw=1.5, alpha=0.8)
    ax.plot([-0.3, 0.9], [-1.1, 0.9], "purple", ls="--", lw=1.5, alpha=0.8)
    ax.text(0.55, 0.75, "linear\nboundaries", color="purple", fontsize=9, ha="center")

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title("Same layout (bad linear probe)\nClasses overlap; no single line splits them")
    ax.axis("off")

    patches = [
        mpatches.Patch(color=colors["cat instances"], label="cat images"),
        mpatches.Patch(color=colors["dog instances"], label="dog images"),
        mpatches.Patch(color=colors["truck instances"], label="truck images"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Physical meaning of good k-NN after InfoNCE: remember *which image*, not *which global class direction*",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
