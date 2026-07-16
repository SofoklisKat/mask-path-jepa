#!/usr/bin/env python3
"""Dummy embedding diagram: why triplet compares masked-prediction vs scramble."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a, b = unit(a), unit(b)
    return 1.0 - float(np.dot(a, b))


def draw_input_icon(ax, kind: str, title: str, subtitle: str) -> None:
    """Simple 8x8 dummy 'image' icons."""
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")
    grid = np.zeros((8, 8, 3))

    if kind == "clean":
        # simple house-like pattern
        grid[2:6, 2:6] = [0.4, 0.7, 0.9]
        grid[1:3, 3:5] = [0.8, 0.5, 0.3]
        grid[5:7, 3:5] = [0.2, 0.6, 0.3]
        ec = "#27ae60"
    elif kind == "corrupt":
        grid[2:6, 2:6] = [0.4, 0.7, 0.9]
        grid[1:3, 3:5] = [0.8, 0.5, 0.3]
        grid[5:7, 3:5] = [0.2, 0.6, 0.3]
        # mask blocks
        for r, c in [(2, 2), (2, 5), (4, 3), (5, 5)]:
            grid[r : r + 2, c : c + 2] = 0
        ec = "#2980b9"
    else:  # scramble
        # 4x4 patches with permuted colors (same palette, wrong places)
        patches = [
            [0.8, 0.5, 0.3],
            [0.4, 0.7, 0.9],
            [0.2, 0.6, 0.3],
            [0.9, 0.8, 0.4],
            [0.3, 0.5, 0.8],
            [0.6, 0.3, 0.7],
            [0.5, 0.8, 0.5],
            [0.7, 0.4, 0.2],
            [0.4, 0.7, 0.9],
            [0.2, 0.6, 0.3],
            [0.8, 0.5, 0.3],
            [0.5, 0.6, 0.9],
            [0.9, 0.7, 0.5],
            [0.3, 0.4, 0.6],
            [0.6, 0.8, 0.4],
            [0.7, 0.5, 0.8],
        ]
        idx = 0
        for r in range(0, 8, 2):
            for c in range(0, 8, 2):
                grid[r : r + 2, c : c + 2] = patches[idx]
                idx += 1
        ec = "#c0392b"

    ax.imshow(grid, extent=(0, 8, 0, 8))
    for x in range(0, 9, 2):
        ax.axvline(x, color="white", lw=0.6, alpha=0.5)
    for y in range(0, 9, 2):
        ax.axhline(y, color="white", lw=0.6, alpha=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold", color=ec)
    ax.text(4, -0.6, subtitle, ha="center", fontsize=9, color=ec, family="monospace")


def draw_embedding_space(ax, good: bool = True) -> None:
    """2D slice of hypersphere with dummy embedding points."""
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")

    # unit circle = hypersphere in 2D
    circle = Circle((0, 0), 1.0, fill=False, linestyle="--", color="#bdc3c7", lw=1.5)
    ax.add_patch(circle)
    ax.text(0, 1.18, "hypersphere (2D slice of 256-D)", ha="center", fontsize=9, color="#7f8c8d")

    # Fixed reference points
    z_clean = unit(np.array([1.0, 0.15]))
    z_scram = unit(np.array([0.35, 0.94]))

    if good:
        # prediction lands near clean, far from scramble
        z_pred = unit(np.array([0.92, 0.28]))
        status = "GOOD: d(pred,clean) < d(pred,scramble) − margin"
        status_color = "#27ae60"
    else:
        # prediction wrongly near scramble
        z_pred = unit(np.array([0.55, 0.82]))
        status = "BAD: model confuses layout — pred near scramble"
        status_color = "#e74c3c"

    points = {
        "z_clean\n(positive)": (z_clean, "#27ae60", "o", 120),
        "z_scramble\n(negative)": (z_scram, "#e74c3c", "X", 130),
        "ẑ_pred\n(from masked)": (z_pred, "#2980b9", "D", 110),
    }

    for label, (p, color, marker, size) in points.items():
        ax.scatter(p[0], p[1], c=color, s=size, marker=marker, zorder=5, edgecolors="white", linewidths=1.2)
        offset = np.array([0.12, 0.12]) if "clean" in label else np.array([-0.15, 0.1])
        if "scramble" in label:
            offset = np.array([-0.05, 0.18])
        if "pred" in label:
            offset = np.array([0.1, -0.18])
        ax.annotate(
            label,
            xy=p,
            xytext=p + offset,
            fontsize=8,
            fontweight="bold",
            color=color,
            ha="center",
        )

    # arcs for distances from pred
    def arc_between(a: np.ndarray, b: np.ndarray, color: str, label: str, rad: float):
        a_n, b_n = unit(a), unit(b)
        angle_a = np.degrees(np.arctan2(a_n[1], a_n[0]))
        angle_b = np.degrees(np.arctan2(b_n[1], b_n[0]))
        # draw chord midpoint label
        mid = unit(a_n + b_n) * 0.72
        d = cosine_dist(a, b)
        ax.plot([a_n[0], b_n[0]], [a_n[1], b_n[1]], color=color, lw=1.5, alpha=0.5, linestyle=":")
        ax.text(mid[0], mid[1], f"{label}\n1−cos={d:.2f}", fontsize=7, color=color, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9))

    arc_between(z_pred, z_clean, "#27ae60", "d_pos", 0.3)
    arc_between(z_pred, z_scram, "#e74c3c", "d_neg", -0.3)

    d_pos = cosine_dist(z_pred, z_clean)
    d_neg = cosine_dist(z_pred, z_scram)
    margin = 0.2
    triplet = max(0, d_pos - d_neg + margin)

    ax.text(
        0,
        -1.32,
        f"triplet = relu(d_pos − d_neg + m) = relu({d_pos:.2f} − {d_neg:.2f} + {margin}) = {triplet:.2f}",
        ha="center",
        fontsize=8,
        family="monospace",
        color="#2c3e50",
    )
    ax.text(0, -1.22, status, ha="center", fontsize=9, fontweight="bold", color=status_color)

    if good:
        ax.text(
            0,
            1.32,
            "Want: masked prediction lands near clean, not near scramble",
            ha="center",
            fontsize=9,
            color="#2c3e50",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/latent_triplet_uniformity/embedding_dummy_explanation.png"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10), facecolor="white")
    fig.suptitle(
        "Why triplet compares masked PREDICTION vs scramble (dummy embeddings)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.2, 0.55], hspace=0.45, wspace=0.3)

    # Row 0: three dummy inputs
    icons = [
        ("clean", "Clean image", "x_clean  →  encoder  →  z_clean"),
        ("corrupt", "Masked image", "x_corrupt  →  enc  →  pred  →  ẑ_pred"),
        ("scramble", "Scrambled (same pixels)", "x_scramble  →  encoder  →  z_scramble"),
    ]
    for col, (kind, title, sub) in enumerate(icons):
        ax = fig.add_subplot(gs[0, col])
        draw_input_icon(ax, kind, title, sub)

    # Row 1: good vs bad embedding layouts
    ax_good = fig.add_subplot(gs[1, 0:2])
    draw_embedding_space(ax_good, good=True)
    ax_good.set_title("What we want in latent space", fontsize=12, fontweight="bold", pad=8)

    ax_bad = fig.add_subplot(gs[1, 2])
    draw_embedding_space(ax_bad, good=False)
    ax_bad.set_title("What we avoid", fontsize=12, fontweight="bold", pad=8)

    # Row 2: explanation text panel
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis("off")
    explanation = """
WHY NOT only compare z_clean vs z_scramble?

  • Clean vs scramble only says: "full correct layout" and "wrong layout" should be far apart in encoder space.
    That does NOT tell the model what to do when the image is MASKED.

  • Our task is JEPA-style: see PART of the image (masked) → PREDICT the full semantic embedding.
    Triplet asks: should that prediction look like the CLEAN embedding, or the SCRAMBLED one?
    Answer must be: like CLEAN (correct global structure), not SCRAMBLE (broken puzzle).

  • Scramble is the perfect label-free negative: SAME image content, WRONG spatial layout.
    If ẑ_pred drifts toward z_scramble, the model is using broken layout cues — bad for class structure.

  • Align loss pulls ẑ_pred → z_clean.  Triplet adds: ẑ_pred must be CLOSER to z_clean than to z_scramble (margin).
"""
    ax_txt.text(
        0.02,
        0.95,
        explanation.strip(),
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#dee2e6"),
    )

    legend_handles = [
        mpatches.Patch(color="#27ae60", label="z_clean — correct layout (positive)"),
        mpatches.Patch(color="#c0392b", label="z_scramble — same image, wrong layout (negative)"),
        mpatches.Patch(color="#2980b9", label="ẑ_pred — prediction from masked view (anchor)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9, frameon=True)

    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
