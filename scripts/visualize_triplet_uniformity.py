#!/usr/bin/env python3
"""Visualize latent_triplet_uniformity: inputs and three loss terms."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch
import torchvision

from tripletjepa.views import block_mask, scramble_patches


def tensor_to_img(x: torch.Tensor) -> np.ndarray:
    """CHW [0,1] -> HWC uint8 for imshow."""
    arr = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(arr, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/latent_triplet_uniformity/triplet_uniformity_losses.png"),
    )
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--patch-size", type=int, default=4)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = torchvision.datasets.CIFAR100(root="./data", train=True, download=True)
    img, _ = ds[args.index]
    x = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)

    clean = x
    corrupt = block_mask(x, mask_ratio=args.mask_ratio)
    scramble = scramble_patches(x, patch_size=args.patch_size)

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.35, wspace=0.25)

    # --- Row 1: three input views ---
    titles = [
        ("Clean (positive view)", "x_clean", "#2ecc71"),
        ("Corrupt (masked, anchor view)", "x_corrupt", "#3498db"),
        ("Scramble (same image, negative)", "x_scramble", "#e74c3c"),
    ]
    views = [clean, corrupt, scramble]
    for col, (title, sym, color) in enumerate(titles):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(tensor_to_img(views[col]))
        ax.set_title(title, fontsize=11, fontweight="bold", color=color)
        ax.axis("off")
        ax.text(
            0.5,
            -0.08,
            sym,
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            color=color,
            family="monospace",
        )

    # --- Row 2: pipeline diagram ---
    ax = fig.add_subplot(gs[1, :])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title(
        "latent_triplet_uniformity — one encoder + predictor (no EMA, no labels)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    def box(x, y, w, h, text, fc, ec, fontsize=9):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color=color,
            )
        )

    # Encoders / predictor boxes
    box(0.3, 4.0, 1.6, 0.9, "encoder", "#ecf0f1", "#7f8c8d", 8)
    box(0.3, 2.5, 1.6, 0.9, "predictor", "#ecf0f1", "#7f8c8d", 8)

    box(2.5, 4.3, 2.0, 0.75, "z_clean\n(stop-grad)", "#d5f5e3", "#27ae60", 8)
    box(2.5, 3.2, 2.0, 0.75, "z_pred = g(enc(corrupt))", "#d6eaf8", "#2980b9", 8)
    box(2.5, 1.8, 2.0, 0.75, "z_scram = enc(scramble)\n(stop-grad)", "#fadbd8", "#c0392b", 8)

    arrow(1.9, 4.45, 2.5, 4.65)
    ax.text(2.15, 4.75, "clean", fontsize=7, color="#27ae60")
    arrow(1.9, 4.45, 2.5, 3.55)
    ax.text(2.0, 3.95, "corrupt", fontsize=7, color="#2980b9")
    arrow(1.9, 2.95, 2.5, 3.35)
    arrow(1.9, 4.45, 2.5, 2.15)
    ax.text(2.0, 2.55, "scramble", fontsize=7, color="#c0392b")

    # Three loss panels
    losses = [
        (
            5.2,
            4.0,
            "① Cosine align (JEPA)",
            "1 − cos(z_pred, z_clean)\nPull predicted latent\ntoward clean encoder",
            "#d5f5e3",
            "#27ae60",
        ),
        (
            5.2,
            2.55,
            "② Cosine triplet",
            "relu(d_pos − d_neg + m)\nd = 1 − cos on sphere\npos=clean, neg=scramble\n(same image, wrong layout)",
            "#fdebd0",
            "#d68910",
        ),
        (
            5.2,
            1.1,
            "③ Uniformity",
            "Spread ẑ on hypersphere\n(encoder corrupt + clean)\nanti-collapse / k-NN geometry",
            "#ebf5fb",
            "#2874a6",
        ),
    ]
    for x0, y0, title, body, fc, ec in losses:
        box(x0, y0, 4.5, 1.25, "", fc, ec)
        ax.text(x0 + 0.15, y0 + 0.95, title, fontsize=10, fontweight="bold", color=ec, va="top")
        ax.text(x0 + 0.15, y0 + 0.72, body, fontsize=8, color="#2c3e50", va="top", family="monospace")

    arrow(4.5, 4.65, 5.2, 4.65, "#27ae60")
    arrow(4.5, 3.55, 5.2, 3.15, "#d68910")
    arrow(4.5, 4.65, 5.2, 1.75, "#2874a6")
    arrow(4.5, 2.15, 5.2, 3.0, "#d68910")

    ax.text(
        5.2,
        0.35,
        "Total:  L = align + λ·triplet + w·uniformity     |     Eval: frozen encoder(clean) only",
        fontsize=9,
        color="#34495e",
        family="monospace",
    )

    legend = [
        mpatches.Patch(color="#d5f5e3", label="positive / alignment"),
        mpatches.Patch(color="#fadbd8", label="negative (label-free distortion)"),
        mpatches.Patch(color="#d6eaf8", label="anchor path (predictor)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)

    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
