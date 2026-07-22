#!/usr/bin/env python3
"""Toy figure: batch InfoNCE with dummy 2D embeddings (educational).

  PYTHONPATH=. python scripts/visualize_infonce_toy.py --output results/analysis/infonce_toy.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

from tripletjepa.losses import infonce_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize batch InfoNCE with dummy inputs")
    parser.add_argument("--output", default="./results/analysis/infonce_toy.png")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    # Dummy batch B=4 on the unit circle (2D for drawing; real model uses D=256).
    # query = masked/aug path, key = clean (detached in training).
    names = ["img A", "img B", "img C", "img D"]
    keys = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-0.8, 0.6],
                [-0.2, -1.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )
    # Queries are noisy versions of matching keys + small cross-talk.
    queries = F.normalize(
        keys
        + torch.tensor(
            [
                [0.15, 0.05],
                [-0.05, 0.12],
                [-0.10, 0.08],
                [0.08, -0.10],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
    )

    loss = infonce_loss(queries, keys, temperature=args.temperature)
    sim = queries @ keys.T
    logits = sim / args.temperature
    probs = F.softmax(logits, dim=-1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 5))

    # Panel 1: geometry on circle
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.set_aspect("equal")
    circle = plt.Circle((0, 0), 1, fill=False, ls="--", color="gray", lw=0.8)
    ax0.add_patch(circle)
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for i, name in enumerate(names):
        kx, ky = keys[i].tolist()
        qx, qy = queries[i].tolist()
        ax0.scatter([kx], [ky], c=colors[i], s=120, marker="o", edgecolors="black", linewidths=0.6)
        ax0.scatter([qx], [qy], c=colors[i], s=80, marker="x", linewidths=2)
        ax0.annotate(name, (kx, ky), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax0.plot([qx, kx], [qy, ky], color=colors[i], alpha=0.5, lw=1.2)
    ax0.set_xlim(-1.3, 1.3)
    ax0.set_ylim(-1.3, 1.3)
    ax0.set_title("Batch on unit circle\n○ = key (clean)  × = query (mask/aug)")
    ax0.grid(True, alpha=0.2)

    # Panel 2: similarity matrix
    ax1 = fig.add_subplot(1, 3, 2)
    im = ax1.imshow(sim.numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax1.set_xticks(range(4), names, rotation=30, ha="right")
    ax1.set_yticks(range(4), [f"q{i}" for i in range(4)])
    for i in range(4):
        for j in range(4):
            ax1.text(j, i, f"{sim[i, j].item():.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax1, fraction=0.046)
    ax1.set_title("Cosine similarity\nquery @ key.T")

    # Panel 3: softmax targets (diagonal = correct)
    ax2 = fig.add_subplot(1, 3, 3)
    im2 = ax2.imshow(probs.numpy(), vmin=0, vmax=1, cmap="Blues")
    ax2.set_xticks(range(4), names, rotation=30, ha="right")
    ax2.set_yticks(range(4), [f"q{i}" for i in range(4)])
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f"{probs[i, j].item():.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im2, ax=ax2, fraction=0.046)
    ax2.set_title(f"Softmax(logits / τ={args.temperature})\nLoss = CE(row, target=i) = {loss.item():.3f}")

    fig.suptitle(
        "InfoNCE (SimCLR-style): each query i must match key i; other keys in batch are negatives",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
